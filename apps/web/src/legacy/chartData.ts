// @ts-nocheck -- scripted prototype code, replaced by the real agent flow in milestone 6
import { rng, seedFrom } from '../lib/format'

// Standard normal CDF and its inverse (Acklam's approximation) for a realistic binormal ROC curve.
function phi(x) {
  const t = 1 / (1 + (0.3275911 * Math.abs(x)) / Math.SQRT2)
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-(x * x) / 2)
  return x >= 0 ? (1 + y) / 2 : (1 - y) / 2
}
function phiInv(p) {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239]
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572]
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
  const q = p < 0.5 ? p : 1 - p
  let x
  if (q < 0.02425) {
    const r = Math.sqrt(-2 * Math.log(q))
    x = (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
  } else {
    const r = q - 0.5,
      s2 = r * r
    x =
      -((((((a[0] * s2 + a[1]) * s2 + a[2]) * s2 + a[3]) * s2 + a[4]) * s2 + a[5]) * r) /
      (((((b[0] * s2 + b[1]) * s2 + b[2]) * s2 + b[3]) * s2 + b[4]) * s2 + 1)
  }
  return p < 0.5 ? -x : x
}

export function rocCurve(auc) {
  const sep = Math.SQRT2 * phiInv(auc)
  return Array.from({ length: 51 }, (_, i) => {
    const fpr = i / 50
    const tpr = fpr === 0 ? 0 : fpr === 1 ? 1 : phi(sep + phiInv(fpr))
    return { fpr: +fpr.toFixed(2), tpr: +tpr.toFixed(3), chance: +fpr.toFixed(2) }
  })
}

export function prCurve(prAuc, prevalence = 0.0017) {
  const m = Math.max(0.5, (1 - prevalence) / (1 - prAuc) - 1)
  return Array.from({ length: 41 }, (_, i) => {
    const recall = i / 40
    return { recall: +recall.toFixed(3), precision: +(1 - (1 - prevalence) * Math.pow(recall, m)).toFixed(3) }
  })
}

export function scoreDistribution(seedKey) {
  const r = rng(seedFrom(seedKey))
  return Array.from({ length: 20 }, (_, i) => {
    const x = (i + 0.5) / 20
    return {
      bin: `${(i / 20).toFixed(2)}`,
      legit: +(100 * Math.exp(-x * 9) * (0.9 + r() * 0.2)).toFixed(2),
      fraud: +(12 * Math.exp((x - 1) * 4) * (0.85 + r() * 0.3)).toFixed(2),
    }
  })
}

export function forecastSeries(seedKey, horizon = 13) {
  const r = rng(seedFrom(seedKey))
  const start = new Date('2025-10-06')
  const histN = 52
  const out = []
  for (let i = 0; i < histN + horizon; i++) {
    const d = new Date(start.getTime() + i * 7 * 86400000)
    const season = Math.sin(((i + 8) / 52) * Math.PI * 2) * 140
    const holiday = i % 52 >= 7 && i % 52 <= 12 ? 260 * Math.sin((((i % 52) - 7) / 5) * Math.PI) : 0
    const trend = 1150 + i * 3.2
    const base = trend + season + holiday
    const label = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
    if (i < histN) {
      const actual = Math.round(base + (r() - 0.5) * 110)
      const fitted = Math.round(base + (r() - 0.5) * 40)
      out.push({ week: label, actual, fitted: i >= histN - 26 ? fitted : null })
    } else {
      const h = i - histN + 1
      const spread = 45 + h * 9
      const f = Math.round(base)
      out.push({ week: label, forecast: f, band: [f - spread, f + spread] })
    }
  }
  // connect the forecast line to the last actual point
  out[histN - 1].forecast = out[histN - 1].actual
  out[histN - 1].band = [out[histN - 1].actual, out[histN - 1].actual]
  return out
}

export function regressionScatter(seedKey, r2 = 0.9) {
  const r = rng(seedFrom(seedKey))
  const noise = Math.sqrt(1 - r2) * 0.55
  return Array.from({ length: 160 }, () => {
    const actual = 140 + Math.pow(r(), 1.8) * 860
    const g = (r() + r() + r() - 1.5) * 1.15
    const predicted = actual * (1 + g * noise)
    return { actual: Math.round(actual), predicted: Math.round(predicted) }
  })
}

export function residualHistogram(seedKey) {
  const r = rng(seedFrom(seedKey) + 7)
  return Array.from({ length: 15 }, (_, i) => {
    const x = (i - 7) / 2.5
    return { bin: `${((i - 7) * 6).toFixed(0)}K`, count: Math.round(420 * Math.exp(-(x * x) / 2) * (0.9 + r() * 0.2)) }
  })
}
