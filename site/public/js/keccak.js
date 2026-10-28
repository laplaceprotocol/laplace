// Keccak-256 as Ethereum uses it: Keccak with a 512-bit capacity and the original padding (0x01 … 0x80), not
// SHA3-256's (0x06). The record's leaves, its Merkle tree and the manifest hash are all keccak-256, so the page can
// check them without trusting anyone's code but this file.
//
// The 1600-bit state is 25 lanes of 64 bits, lane x + 5y as in FIPS 202. JavaScript's bitwise operators work on
// 32 bits, so each lane is kept as two halves, `lo` and `hi`.

const RATE = 136; // bytes absorbed per permutation: (1600 - 2 * 256) / 8
const ROUNDS = 24;

// The round constants of ι (FIPS 202, 3.2.5), high and low halves.
const RC_HI = Uint32Array.of(
  0x00000000, 0x00000000, 0x80000000, 0x80000000, 0x00000000, 0x00000000, 0x80000000, 0x80000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x80000000, 0x80000000, 0x80000000,
  0x80000000, 0x80000000, 0x00000000, 0x80000000, 0x80000000, 0x80000000, 0x00000000, 0x80000000,
);
const RC_LO = Uint32Array.of(
  0x00000001, 0x00008082, 0x0000808a, 0x80008000, 0x0000808b, 0x80000001, 0x80008081, 0x00008009,
  0x0000008a, 0x00000088, 0x80008009, 0x8000000a, 0x8000808b, 0x0000008b, 0x00008089, 0x00008003,
  0x00008002, 0x00000080, 0x0000800a, 0x8000000a, 0x80008081, 0x00008080, 0x80000001, 0x80008008,
);

// ρ and π (FIPS 202, 3.2.2 and 3.2.3): lane (x, y) is rotated by RHO[x + 5y] bits and moves to PI[x + 5y], which is
// lane (y, 2x + 3y). The offsets follow the walk from lane (1, 0): the t-th lane visited rotates by (t + 1)(t + 2) / 2.
const RHO = new Uint8Array(25);
const PI = new Uint8Array(25);
for (let t = 0, x = 1, y = 0; t < 24; t++) {
  RHO[x + 5 * y] = (((t + 1) * (t + 2)) / 2) % 64;
  [x, y] = [y, (2 * x + 3 * y) % 5];
}
for (let x = 0; x < 5; x++) for (let y = 0; y < 5; y++) PI[x + 5 * y] = y + 5 * ((2 * x + 3 * y) % 5);

function permute(lo, hi) {
  const cLo = new Uint32Array(5);
  const cHi = new Uint32Array(5);
  const bLo = new Uint32Array(25);
  const bHi = new Uint32Array(25);
  for (let round = 0; round < ROUNDS; round++) {
    // θ: every lane takes the parity of two neighbouring columns, one of them rotated by a bit.
    for (let x = 0; x < 5; x++) {
      cLo[x] = lo[x] ^ lo[x + 5] ^ lo[x + 10] ^ lo[x + 15] ^ lo[x + 20];
      cHi[x] = hi[x] ^ hi[x + 5] ^ hi[x + 10] ^ hi[x + 15] ^ hi[x + 20];
    }
    for (let x = 0; x < 5; x++) {
      const before = (x + 4) % 5;
      const after = (x + 1) % 5;
      const dLo = cLo[before] ^ ((cLo[after] << 1) | (cHi[after] >>> 31));
      const dHi = cHi[before] ^ ((cHi[after] << 1) | (cLo[after] >>> 31));
      for (let y = 0; y < 25; y += 5) {
        lo[x + y] ^= dLo;
        hi[x + y] ^= dHi;
      }
    }
    // ρ and π: rotate every lane and move it.
    for (let i = 0; i < 25; i++) {
      const n = RHO[i];
      const l = lo[i];
      const h = hi[i];
      if (n === 0) {
        bLo[PI[i]] = l;
        bHi[PI[i]] = h;
      } else if (n < 32) {
        bLo[PI[i]] = (l << n) | (h >>> (32 - n));
        bHi[PI[i]] = (h << n) | (l >>> (32 - n));
      } else if (n === 32) {
        bLo[PI[i]] = h;
        bHi[PI[i]] = l;
      } else {
        bLo[PI[i]] = (h << (n - 32)) | (l >>> (64 - n));
        bHi[PI[i]] = (l << (n - 32)) | (h >>> (64 - n));
      }
    }
    // χ: the only non-linear step, along each row.
    for (let y = 0; y < 25; y += 5) {
      for (let x = 0; x < 5; x++) {
        const next = ((x + 1) % 5) + y;
        const second = ((x + 2) % 5) + y;
        lo[x + y] = bLo[x + y] ^ (~bLo[next] & bLo[second]);
        hi[x + y] = bHi[x + y] ^ (~bHi[next] & bHi[second]);
      }
    }
    // ι
    lo[0] ^= RC_LO[round];
    hi[0] ^= RC_HI[round];
  }
}

/** keccak256 of `bytes` (a Uint8Array), as 32 bytes. */
export function keccak256(bytes) {
  const blocks = Math.floor(bytes.length / RATE) + 1; // the padding always takes at least one byte
  const padded = new Uint8Array(blocks * RATE);
  padded.set(bytes);
  padded[bytes.length] ^= 0x01;
  padded[padded.length - 1] ^= 0x80;

  const lo = new Uint32Array(25);
  const hi = new Uint32Array(25);
  const view = new DataView(padded.buffer);
  for (let offset = 0; offset < padded.length; offset += RATE) {
    for (let i = 0; i < RATE / 8; i++) {
      lo[i] ^= view.getUint32(offset + 8 * i, true);
      hi[i] ^= view.getUint32(offset + 8 * i + 4, true);
    }
    permute(lo, hi);
  }

  const out = new Uint8Array(32);
  const result = new DataView(out.buffer);
  for (let i = 0; i < 4; i++) {
    result.setUint32(8 * i, lo[i], true);
    result.setUint32(8 * i + 4, hi[i], true);
  }
  return out;
}
