import { describe, expect, it } from 'vitest'
import { bytes, compact, pct, timeAgo } from '../../src/lib/format'

describe('format helpers', () => {
  it('compacts large numbers', () => {
    expect(compact(18412)).toBe('18K')
    expect(compact(1250)).toBe('1.3K')
    expect(compact(2_500_000)).toBe('2.50M')
  })
  it('formats percentages and bytes', () => {
    expect(pct(0.1234)).toBe('12.3%')
    expect(bytes(2048)).toBe('2 KB')
  })
  it('describes recent times', () => {
    expect(timeAgo(Date.now() - 5_000)).toBe('just now')
    expect(timeAgo(Date.now() - 3 * 60_000)).toBe('3 min ago')
  })
})
