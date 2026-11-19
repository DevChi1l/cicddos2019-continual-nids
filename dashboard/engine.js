// Tabular Transformer forward pass in plain JavaScript (no libraries).
// Mirrors s2_model.TabTransformer: tokens [CLS, 77 numeric, protocol], 3 x nn.TransformerEncoderLayer
// (norm_first=True, GELU, 4 heads), LayerNorm on [CLS], Linear head. Dropout is off (eval mode).
"use strict";
var NIDS = (function () {
  function erf(x) {
    // Numerical Recipes erfc (Chebyshev), fractional error < 1.2e-7
    var z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    var r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
      t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
      t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? 1 - r : r - 1;
  }

  function scaleRow(raw, sc) {
    var F = sc.features_numeric.length, x = new Float64Array(F), c = sc.clip;
    for (var j = 0; j < F; j++) {
      var v = raw[j], s = (v < 0 ? -1 : v > 0 ? 1 : 0) * Math.log1p(Math.abs(v));
      var z = (s - sc.mean[j]) / sc.std[j];
      if (z > c) z = c; else if (z < -c) z = -c;
      x[j] = Math.fround(z);
    }
    return x;
  }

  function protoCode(p, sc) {
    var voc = sc.cat_vocab[sc.features_categorical[0]];
    var k = String(Math.trunc(p));
    return voc.hasOwnProperty(k) ? voc[k] : 0;
  }

  function Model(m, hp) {
    var D = hp.d, H = hp.heads, L = hp.layers, dh = D / H;
    var t = m.tensors;
    function get(n) { return Float64Array.from(t[n].data); }
    var W = get("tokenizer.W"), B = get("tokenizer.b"), CLS = get("tokenizer.cls"), CAT = get("tokenizer.cat_emb.0.weight");
    var F = t["tokenizer.W"].shape[0], T = F + 2, FF = t["encoder.layers.0.linear1.weight"].shape[0];
    var layers = [];
    for (var l = 0; l < L; l++) {
      var p = "encoder.layers." + l + ".";
      layers.push({
        n1w: get(p + "norm1.weight"), n1b: get(p + "norm1.bias"), n2w: get(p + "norm2.weight"), n2b: get(p + "norm2.bias"),
        inW: get(p + "self_attn.in_proj_weight"), inB: get(p + "self_attn.in_proj_bias"),
        oW: get(p + "self_attn.out_proj.weight"), oB: get(p + "self_attn.out_proj.bias"),
        l1W: get(p + "linear1.weight"), l1B: get(p + "linear1.bias"), l2W: get(p + "linear2.weight"), l2B: get(p + "linear2.bias")
      });
    }
    var nW = get("norm.weight"), nB = get("norm.bias"), hW = get("head.weight"), hB = get("head.bias");
    var K = t["head.weight"].shape[0];
    // work buffers
    var h = new Float64Array(T * D), z = new Float64Array(T * D), qkv = new Float64Array(T * 3 * D),
      att = new Float64Array(T * D), sc = new Float64Array(T), u = new Float64Array(FF), e = new Float64Array(D);
    var invs = 1 / Math.sqrt(dh);

    function layerNorm(src, off, w, b, dst, doff) {
      var m = 0, v = 0, i;
      for (i = 0; i < D; i++) m += src[off + i];
      m /= D;
      for (i = 0; i < D; i++) { var d0 = src[off + i] - m; v += d0 * d0; }
      var r = 1 / Math.sqrt(v / D + 1e-5);
      for (i = 0; i < D; i++) dst[doff + i] = (src[off + i] - m) * r * w[i] + b[i];
    }

    function forward(x, code) {
      var i, j, k, tt, s;
      for (i = 0; i < D; i++) h[i] = CLS[i];
      for (j = 0; j < F; j++) { var o = (j + 1) * D, xv = x[j]; for (i = 0; i < D; i++) h[o + i] = xv * W[j * D + i] + B[j * D + i]; }
      var oc = (F + 1) * D;
      for (i = 0; i < D; i++) h[oc + i] = CAT[code * D + i];
      for (var l = 0; l < L; l++) {
        var Ly = layers[l];
        for (tt = 0; tt < T; tt++) layerNorm(h, tt * D, Ly.n1w, Ly.n1b, z, tt * D);
        // q, k, v projections
        for (tt = 0; tt < T; tt++) {
          var zo = tt * D, qo = tt * 3 * D;
          for (k = 0; k < 3 * D; k++) {
            s = Ly.inB[k]; var wo = k * D;
            for (i = 0; i < D; i++) s += Ly.inW[wo + i] * z[zo + i];
            qkv[qo + k] = s;
          }
        }
        // attention per head
        for (var hd = 0; hd < H; hd++) {
          var ho = hd * dh;
          for (tt = 0; tt < T; tt++) {
            var qb = tt * 3 * D + ho, mx = -Infinity;
            for (j = 0; j < T; j++) {
              var kb = j * 3 * D + D + ho; s = 0;
              for (i = 0; i < dh; i++) s += qkv[qb + i] * qkv[kb + i];
              s *= invs; sc[j] = s; if (s > mx) mx = s;
            }
            var sum = 0;
            for (j = 0; j < T; j++) { sc[j] = Math.exp(sc[j] - mx); sum += sc[j]; }
            for (i = 0; i < dh; i++) {
              s = 0;
              for (j = 0; j < T; j++) s += sc[j] * qkv[j * 3 * D + 2 * D + ho + i];
              att[tt * D + ho + i] = s / sum;
            }
          }
        }
        // out projection + residual
        for (tt = 0; tt < T; tt++) {
          var ao = tt * D;
          for (k = 0; k < D; k++) {
            s = Ly.oB[k]; var wo2 = k * D;
            for (i = 0; i < D; i++) s += Ly.oW[wo2 + i] * att[ao + i];
            h[ao + k] += s;
          }
        }
        // feed-forward + residual
        for (tt = 0; tt < T; tt++) {
          var to = tt * D;
          layerNorm(h, to, Ly.n2w, Ly.n2b, z, to);
          for (k = 0; k < FF; k++) {
            s = Ly.l1B[k]; var w1 = k * D;
            for (i = 0; i < D; i++) s += Ly.l1W[w1 + i] * z[to + i];
            u[k] = 0.5 * s * (1 + erf(s * Math.SQRT1_2));
          }
          for (k = 0; k < D; k++) {
            s = Ly.l2B[k]; var w2 = k * FF;
            for (i = 0; i < FF; i++) s += Ly.l2W[w2 + i] * u[i];
            h[to + k] += s;
          }
        }
      }
      layerNorm(h, 0, nW, nB, e, 0);
      var lo = new Float64Array(K), mx2 = -Infinity;
      for (k = 0; k < K; k++) { s = hB[k]; for (i = 0; i < D; i++) s += hW[k * D + i] * e[i]; lo[k] = s; if (s > mx2) mx2 = s; }
      var tot = 0;
      for (k = 0; k < K; k++) { lo[k] = Math.exp(lo[k] - mx2); tot += lo[k]; }
      for (k = 0; k < K; k++) lo[k] /= tot;
      return lo;
    }
    return { classes: m.classes, forward: forward, nFeatures: F };
  }

  return { erf: erf, scaleRow: scaleRow, protoCode: protoCode, Model: Model };
})();
if (typeof module !== "undefined") module.exports = NIDS;
