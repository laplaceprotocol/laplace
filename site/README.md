# site

The public page at `laplaceprotocol.com`: the latest issue's odds, its commitment, the scores with each rule's running
statistic, how to verify an issue, and the corrections log. It is static. In the browser it reads only the public
record (`https://data.laplaceprotocol.com/v1/`) and Robinhood Chain's public RPC endpoints, and it checks an issue
there with its own keccak-256, Merkle tree and pool mixing, which mirror the engine's (`engine/src/laplace_engine/
record.py`, `quantize.py`, `verify.py`) and are tested against the shared vectors in `contracts/test/vectors`.

| file | |
|---|---|
| `public/index.html`, `style.css` | the page and its static copy (the corrections log mirrors [`docs/corrections.md`](../docs/corrections.md)) |
| `public/js/keccak.js`, `abi.js` | keccak-256 and the bits of ABI encoding the record uses |
| `public/js/record.js` | leaves, the Merkle tree, proofs, pool mixing, program identities, rebuilding an issue |
| `public/js/chain.js` | `rootOf`, the `Committed` event and `verifyForecast`, read from public RPC endpoints |
| `public/js/odds.js`, `status.js`, `scores.js`, `check.js` | the sections; `content.js` holds the rules' terms and the ladder, copied from the docs they link to |
| `public/_headers` | the Content-Security-Policy: scripts and styles from the page only, connections to the record and the RPCs only |

```bash
npm test                                   # node --test, no dependencies
npx wrangler@4 dev                         # http://localhost:8787/, with the headers it will be served with
```

The page reads the live record from the browser, so a local copy only works once the record answers other origins
(step 1 below).

## Deploying
Not deployed yet. It is a Worker with static assets and no code, on the Laplace account. Before the first deploy:

1. **CORS on the record.** The page fetches the record from another origin, so the bucket behind
   `data.laplaceprotocol.com` must answer `GET` and `HEAD` from any origin (`Access-Control-Allow-Origin: *`; the
   record is public). Check that cached files carry the header too:
   `curl -sI -H 'Origin: https://laplaceprotocol.com' https://data.laplaceprotocol.com/v1/issues/498103/manifest.json`.
2. **Deploy.** `CLOUDFLARE_ACCOUNT_ID=… CLOUDFLARE_API_TOKEN=… npx wrangler@4 deploy` from this directory. The custom
   domain route creates the `laplaceprotocol.com` DNS record on the first deploy.
