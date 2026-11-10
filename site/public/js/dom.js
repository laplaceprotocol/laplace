// A minimal element builder. Text is always set as text, never parsed as HTML, so nothing from the record can
// inject markup into the page.

const SVG = "http://www.w3.org/2000/svg";
const SVG_TAGS = new Set(["svg", "g", "rect", "line", "text", "title", "path"]);

/** h("a", { href, class: "x" }, "text", child, …): attributes, then children (strings, nodes, arrays, or null). */
export function h(tag, attrs = {}, ...children) {
  const el = SVG_TAGS.has(tag) ? document.createElementNS(SVG, tag) : document.createElement(tag);
  for (const [key, value] of Object.entries(attrs ?? {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : String(value));
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(typeof child === "string" || typeof child === "number" ? String(child) : child);
  }
}

/** Replace an element's children. */
export function fill(el, ...children) {
  el.replaceChildren();
  append(el, children);
  return el;
}

export function link(href, text, attrs = {}) {
  const external = /^https?:/.test(href);
  return h("a", { href, ...(external ? { rel: "noopener" } : {}), ...attrs }, text);
}
