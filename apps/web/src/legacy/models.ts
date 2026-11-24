// @ts-nocheck -- scripted prototype code, replaced by the real agent flow in milestone 6
import { SCENARIOS } from './scenarios'
import { rng } from '../lib/format'

const DAY = 86400000

// Two extra specs that only exist in the catalog (not offered as agent scripts).
export const EXTRA_SPECS = {
  segments: {
    key: 'segments',
    kind: 'clustering',
    problemType: 'Clustering',
    title: 'Customer Segmentation (RFM+)',
    slug: 'customer_segments_rfm',
    goal: 'Group our customers into actionable segments for marketing',
    source: {
      connector: 'Snowflake',
      connectorId: 'snowflake',
      database: 'PROD_ANALYTICS',
      tables: [
        { name: 'CUSTOMERS', role: 'Primary entity', rows: 212400, join: 'customer_id [PK]' },
        { name: 'ORDERS', role: 'Transactions', rows: 1840221, join: 'customer_id [FK]' },
        { name: 'WEB_EVENTS', role: 'Behaviour', rows: 22410337, join: 'customer_id [as-of]' },
      ],
    },
    dataset: {
      rows: 212400,
      cols: 24,
      period: 'Oct 2024 – Sep 2026',
      quality: 90,
      entity: 'customer_id',
      target: '— (unsupervised)',
      balance: '5 segments discovered',
    },
    schema: [
      { name: 'customer_id', type: 'STRING', role: 'Entity key', missing: 0, unique: '212,400' },
      { name: 'recency_days', type: 'INT', role: 'Feature', missing: 0, unique: '730' },
      { name: 'frequency_12m', type: 'INT', role: 'Feature', missing: 0, unique: '96' },
      { name: 'monetary_12m', type: 'FLOAT', role: 'Feature', missing: 0, unique: '61,022' },
      { name: 'avg_basket_size', type: 'FLOAT', role: 'Feature', missing: 0, unique: '8,410' },
      { name: 'discount_share', type: 'FLOAT', role: 'Feature', missing: 1.2, unique: '1,004' },
      { name: 'channel_mix_web', type: 'FLOAT', role: 'Feature', missing: 0, unique: '101' },
    ],
    strategy: {
      metric: 'Silhouette',
      validation: 'Bootstrap stability (ARI) over 20 resamples',
      extras: [
        ['Scaling', 'Robust scaler + log monetary'],
        ['k search', 'k = 3 … 8'],
      ],
    },
    cleaning: [
      { issue: 'Monetary heavily skewed', action: 'log1p transform before clustering', count: 'all rows' },
      { issue: '2,840 B2B reseller accounts', action: 'Excluded (separate segment strategy)', count: '2,840 rows' },
    ],
    features: [
      { name: 'recency_days', desc: 'Days since last order', genai: false },
      { name: 'frequency_12m', desc: 'Orders in 12 months', genai: false },
      { name: 'monetary_12m_log', desc: 'Log spend', genai: false },
      { name: 'discount_share', desc: 'Share of orders with discount', genai: true },
      { name: 'category_breadth', desc: 'Distinct categories bought', genai: true },
    ],
    featureCount: 18,
    metricName: 'Silhouette',
    better: 'higher',
    fmt: (v) => v.toFixed(2),
    candidates: [
      { name: 'K-Means (k=4)', family: 'Centroid', r1: 0.52, r2: 0.52, time: '22s' },
      { name: 'K-Means (k=5)', family: 'Centroid', r1: 0.58, r2: 0.58, time: '25s' },
      { name: 'K-Means (k=6)', family: 'Centroid', r1: 0.55, r2: 0.55, time: '29s' },
      { name: 'Gaussian Mixture', family: 'Probabilistic', r1: 0.53, r2: 0.53, time: '1m 10s' },
      { name: 'HDBSCAN', family: 'Density', r1: 0.49, r2: 0.49, time: '2m 40s' },
    ],
    critic: {
      findings: [
        { severity: 'pass', title: 'Segments are stable', detail: 'Adjusted Rand Index 0.92 across 20 bootstrap resamples.' },
        { severity: 'low', title: 'One small segment', detail: 'Champions is 12%, but it is still large enough to act on.' },
      ],
    },
    final: {
      algorithm: 'K-Means (k = 5)',
      primary: { name: 'Silhouette', display: '0.58' },
      baseline: { name: 'K-Means (k = 3)', display: '0.41' },
      metrics: [
        { name: 'Silhouette', display: '0.58', baseline: '0.41', good: true },
        { name: 'Davies–Bouldin', display: '0.71', baseline: '1.08', good: true },
        { name: 'Stability (ARI)', display: '0.92', baseline: '—', good: true },
        { name: 'Segments', display: '5', baseline: '—', good: true },
      ],
      importance: [
        ['monetary_12m_log', 0.31],
        ['recency_days', 0.27],
        ['frequency_12m', 0.2],
        ['discount_share', 0.13],
        ['category_breadth', 0.09],
      ],
      explanation:
        'Customers split naturally into five groups by how recently, how often and how much they buy. The biggest opportunity is "At Risk" (27%): customers who used to buy often but haven\'t ordered in 90+ days.',
      whyModel:
        'k = 5 had the best silhouette (0.58) and gave segments marketing could name and act on. k = 6 split "Loyal" into two near-identical groups.',
      segments: [
        { name: 'Champions', share: 12, desc: 'Recent, frequent, high spend' },
        { name: 'Loyal', share: 24, desc: 'Frequent, mid spend' },
        { name: 'Promising', share: 19, desc: 'New, growing basket' },
        { name: 'At Risk', share: 27, desc: 'Was frequent, now 90+ days silent' },
        { name: 'Hibernating', share: 18, desc: 'Low recency & spend' },
      ],
    },
    decisions: [
      {
        stage: 'detect',
        step: 'Problem detection',
        decision: 'Clustering (unsupervised)',
        reason: 'Goal asks to "group customers"; no target column.',
        alternatives: 'Rule-based RFM bins',
      },
      {
        stage: 'clean',
        step: 'Cleaning',
        decision: 'Excluded B2B resellers',
        reason: 'Order patterns 40× larger; would dominate clusters.',
        alternatives: 'Keep all',
      },
      {
        stage: 'train',
        step: 'Model search',
        decision: 'Searched k = 3…8 + GMM + HDBSCAN',
        reason: 'Compare centroid, probabilistic and density views.',
        alternatives: 'Fixed k = 4',
      },
      {
        stage: 'final',
        step: 'Model selection',
        decision: 'K-Means, k = 5',
        reason: 'Best silhouette + business interpretability.',
        alternatives: 'k = 6, GMM',
      },
    ],
    deploy: {
      path: 'assign',
      latency: 12,
      sampleInput: { customer_id: 'c_120044', recency_days: 112, frequency_12m: 9, monetary_12m: 1840 },
      sampleOutput: { segment: 'At Risk', distance_to_centroid: 0.42 },
    },
  },
  loan: {
    key: 'loan',
    kind: 'classification',
    problemType: 'Binary Classification',
    title: 'Loan Default Risk',
    slug: 'loan_default_risk',
    goal: 'Predict whether a borrower will default within 12 months',
    source: {
      connector: 'PostgreSQL',
      connectorId: 'postgres',
      database: 'lending_core',
      tables: [
        { name: 'LOANS', role: 'Primary entity', rows: 86210, join: 'loan_id [PK]' },
        { name: 'BORROWERS', role: 'Applicant profile', rows: 79404, join: 'borrower_id [FK]' },
        { name: 'REPAYMENTS', role: 'Temporal events', rows: 1902114, join: 'loan_id [as-of]' },
      ],
    },
    dataset: {
      rows: 86210,
      cols: 41,
      period: 'Jan 2022 – Sep 2026',
      quality: 85,
      entity: 'loan_id',
      target: 'defaulted_12m',
      balance: '6.4% default · 93.6% repaid',
    },
    schema: [
      { name: 'loan_id', type: 'STRING', role: 'Entity key', missing: 0, unique: '86,210' },
      { name: 'loan_amount', type: 'FLOAT', role: 'Feature', missing: 0, unique: '12,044' },
      { name: 'interest_rate', type: 'FLOAT', role: 'Feature', missing: 0, unique: '402' },
      { name: 'debt_to_income', type: 'FLOAT', role: 'Feature', missing: 1.9, unique: '5,881' },
      { name: 'credit_score', type: 'INT', role: 'Feature', missing: 0.7, unique: '421' },
      { name: 'employment_years', type: 'INT', role: 'Feature', missing: 4.4, unique: '41' },
      { name: 'defaulted_12m', type: 'BOOL', role: 'Target', missing: 0, unique: '2' },
    ],
    strategy: {
      metric: 'ROC-AUC',
      validation: 'Out-of-time: train 2022–2025, test 2026',
      extras: [
        ['Fairness check', 'Equal opportunity across age bands'],
        ['Imbalance', 'Class weights 1 : 14'],
      ],
    },
    cleaning: [
      { issue: 'employment_years missing 4.4%', action: 'Imputed + missing flag', count: '3,793 cells' },
      { issue: 'Loans < 12 months old', action: 'Excluded (label not mature)', count: '9,120 rows' },
    ],
    features: [
      { name: 'debt_to_income', desc: 'Debt / income', genai: false },
      { name: 'payment_to_income', desc: 'Instalment / monthly income', genai: true },
      { name: 'credit_score', desc: 'Bureau score', genai: false },
      { name: 'late_payments_6m', desc: 'Late payments in 6 months', genai: false },
    ],
    featureCount: 39,
    metricName: 'ROC-AUC',
    better: 'higher',
    fmt: (v) => v.toFixed(3),
    candidates: [
      { name: 'Logistic Regression', family: 'Linear', r1: 0.811, r2: 0.811, time: '9s' },
      { name: 'LightGBM', family: 'Boosting', r1: 0.866, r2: 0.866, time: '58s' },
      { name: 'XGBoost', family: 'Boosting', r1: 0.874, r2: 0.874, time: '1m 40s' },
      { name: 'CatBoost', family: 'Boosting', r1: 0.869, r2: 0.869, time: 'running…' },
    ],
    critic: { findings: [{ severity: 'medium', title: 'Fairness review pending', detail: 'Equal-opportunity check runs after HPO completes.' }] },
    final: {
      algorithm: 'XGBoost (in search)',
      primary: { name: 'ROC-AUC', display: '0.874' },
      baseline: { name: 'Credit score only', display: '0.742' },
      metrics: [
        { name: 'ROC-AUC (interim)', display: '0.874', baseline: '0.742', good: true },
        { name: 'PR-AUC (interim)', display: '0.412', baseline: '0.064', good: true },
        { name: 'KS statistic', display: '0.58', baseline: '0.38', good: true },
      ],
      importance: [
        ['payment_to_income', 0.23],
        ['credit_score', 0.21],
        ['late_payments_6m', 0.18],
        ['debt_to_income', 0.14],
        ['interest_rate', 0.1],
      ],
      explanation:
        'Interim result: borrowers whose instalment is a large share of income and who have recent late payments are most likely to default.',
      whyModel: 'Search in progress. XGBoost currently leads, with CatBoost still training.',
      confusion: { tp: 812, fn: 290, fp: 2410, tn: 13730 },
      auc: 0.874,
    },
    decisions: [
      {
        stage: 'detect',
        step: 'Problem detection',
        decision: 'Binary classification',
        reason: 'Boolean target defaulted_12m.',
        alternatives: 'Survival analysis',
      },
      {
        stage: 'clean',
        step: 'Cleaning',
        decision: 'Excluded loans younger than 12 months',
        reason: 'Their outcome is not yet known.',
        alternatives: 'Treat as non-default (biased)',
      },
      {
        stage: 'train',
        step: 'Model search',
        decision: 'HPO budget 120 trials',
        reason: 'Regulated use case: worth extra compute.',
        alternatives: '40 trials',
      },
    ],
    deploy: {
      path: 'predict',
      latency: 22,
      sampleInput: { loan_id: 'L-30021', loan_amount: 18000, credit_score: 662, debt_to_income: 0.41 },
      sampleOutput: { default_probability: 0.18, risk_grade: 'C' },
    },
  },
}

export function getSpec(key) {
  return SCENARIOS[key] || EXTRA_SPECS[key]
}

export function bestCandidate(spec) {
  const dir = spec.better === 'lower' ? 1 : -1
  return [...spec.candidates].sort((a, b) => dir * (a.r2 - b.r2))[0]
}

function usageSeries(seed, base) {
  const r = rng(seed)
  return Array.from({ length: 30 }, (_, i) => {
    const weekday = i % 7 < 5 ? 1 : 0.72
    return { day: i, value: Math.round(base * weekday * (0.85 + r() * 0.3) * (1 + i * 0.006)) }
  })
}

export function buildModel(spec, o = {}) {
  const now = Date.now()
  const best = bestCandidate(spec)
  const pred30 = o.predictions30d ?? 0
  return {
    id: o.id || spec.slug,
    specKey: spec.key,
    name: o.name || spec.title,
    slug: o.slug || spec.slug,
    kind: spec.kind,
    problemType: spec.problemType,
    objective: o.objective || spec.goal,
    status: o.status || 'live',
    progress: o.progress ?? 100,
    stage: o.stage || 'Deployed',
    version: o.version || 'v1.0',
    owner: 'AI Data Scientist',
    requestedBy: o.requestedBy || 'Akash Kesarkar',
    createdAt: o.createdAt || now,
    updatedAt: o.updatedAt || now,
    createdBy: o.createdBy || 'seed',
    algorithm: spec.final.algorithm,
    primary: spec.final.primary,
    baseline: spec.final.baseline,
    metrics: spec.final.metrics,
    metricName: spec.metricName,
    better: spec.better,
    source: spec.source,
    dataset: spec.dataset,
    schema: spec.schema,
    strategy: { metric: spec.strategy.metric, validation: spec.strategy.validation, extras: spec.strategy.extras },
    cleaning: spec.cleaning,
    features: spec.features,
    featureCount: spec.featureCount,
    candidates: spec.candidates.map((c) => ({
      name: c.name,
      family: c.family,
      score: c.r2,
      display: spec.fmt(c.r2),
      time: c.time,
      selected: c === best,
    })),
    critic: spec.critic.findings,
    importance: spec.final.importance,
    explanation: spec.final.explanation,
    whyModel: spec.final.whyModel,
    confusion: spec.final.confusion || null,
    auc: spec.final.auc || null,
    r2: spec.final.r2 || null,
    backtest: spec.final.backtest || null,
    segments: spec.final.segments || null,
    decisions: (o.decisions || spec.decisions).map(({ step, decision, reason, alternatives }) => ({
      step,
      decision,
      reason,
      alternatives: alternatives || '',
    })),
    endpoint: `https://api.datalyon.ai/v1/models/${o.slug || spec.slug}/${spec.deploy.path}`,
    sampleInput: spec.deploy.sampleInput,
    sampleOutput: spec.deploy.sampleOutput,
    deployment: o.deployment || 'REST API · real-time',
    usage: {
      predictions30d: pred30,
      latency: spec.deploy.latency,
      uptime: o.status === 'live' ? '99.98%' : '—',
      series: pred30 ? usageSeries(spec.slug.length * 7 + pred30, pred30 / 30) : [],
    },
    drift: o.drift || { status: 'stable', value: 0.04, note: 'No significant input drift (PSI < 0.1)' },
    versions: o.versions || [
      {
        version: o.version || 'v1.0',
        date: o.createdAt || now,
        metric: spec.final.primary.display,
        note: 'Initial build by AI Data Scientist',
        status: o.status || 'live',
      },
    ],
    tags: o.tags || [],
  }
}

export function seedModels() {
  const now = Date.now()
  return [
    buildModel(SCENARIOS.churn, {
      id: 'churn_prod',
      name: '30-Day Customer Churn',
      slug: 'churn_prod_v3',
      status: 'live',
      version: 'v3.2',
      createdAt: now - 142 * DAY,
      updatedAt: now - 3 * DAY,
      predictions30d: 412300,
      tags: ['customer-success', 'weekly-batch'],
      deployment: 'Batch · daily 06:00 UTC + REST API',
      versions: [
        { version: 'v3.2', date: now - 3 * DAY, metric: '0.912', note: 'Agent retrain: added payment failure features', status: 'live' },
        { version: 'v3.1', date: now - 46 * DAY, metric: '0.897', note: 'Monthly scheduled retrain', status: 'archived' },
        { version: 'v2.0', date: now - 98 * DAY, metric: '0.884', note: 'Switched XGBoost → LightGBM', status: 'archived' },
        { version: 'v1.0', date: now - 142 * DAY, metric: '0.861', note: 'Initial build by AI Data Scientist', status: 'archived' },
      ],
    }),
    buildModel(SCENARIOS.forecast, {
      id: 'demand_prod',
      name: 'Weekly Store Demand Forecast',
      slug: 'store_demand_weekly',
      status: 'live',
      version: 'v2.4',
      createdAt: now - 201 * DAY,
      updatedAt: now - 1 * DAY,
      predictions30d: 18720,
      tags: ['supply-chain'],
      deployment: 'Batch · weekly Monday 02:00 UTC',
      versions: [
        { version: 'v2.4', date: now - 1 * DAY, metric: '6.2%', note: 'Critic loop: holiday interaction features', status: 'live' },
        { version: 'v2.3', date: now - 33 * DAY, metric: '7.1%', note: 'Added promotions as drivers', status: 'archived' },
        { version: 'v1.0', date: now - 201 * DAY, metric: '9.4%', note: 'Initial Prophet baseline', status: 'archived' },
      ],
    }),
    buildModel(SCENARIOS.fraud, {
      id: 'fraud_prod',
      name: 'Card Transaction Fraud Detector',
      slug: 'txn_fraud_detector',
      status: 'live',
      version: 'v4.0',
      createdAt: now - 88 * DAY,
      updatedAt: now - 9 * DAY,
      predictions30d: 6312000,
      tags: ['real-time', 'risk'],
      drift: { status: 'warning', value: 0.17, note: 'merchant_category drift (PSI 0.17). Retrain suggested.' },
    }),
    buildModel(SCENARIOS.price, {
      id: 'price_staging',
      name: 'Home Price Estimator',
      slug: 'home_price_estimator',
      status: 'staging',
      version: 'v1.1',
      createdAt: now - 12 * DAY,
      updatedAt: now - 2 * DAY,
      predictions30d: 2140,
      tags: ['pricing'],
      deployment: 'REST API · staging',
    }),
    buildModel(EXTRA_SPECS.segments, {
      id: 'segments_prod',
      status: 'live',
      version: 'v1.3',
      createdAt: now - 64 * DAY,
      updatedAt: now - 6 * DAY,
      predictions30d: 212400,
      tags: ['marketing'],
      deployment: 'Batch · nightly',
    }),
    buildModel(EXTRA_SPECS.loan, {
      id: 'loan_training',
      status: 'training',
      progress: 64,
      stage: 'Model search + HPO',
      version: 'v0.9',
      createdAt: now - 0.2 * DAY,
      updatedAt: now - 0.01 * DAY,
      predictions30d: 0,
      tags: ['regulated'],
      deployment: 'Not deployed',
    }),
  ]
}

export function seedConnectors() {
  const now = Date.now()
  return [
    {
      id: 'snowflake',
      name: 'Snowflake',
      detail: 'PROD_ANALYTICS · PAYMENTS',
      status: 'connected',
      tables: 42,
      volume: '1.2B rows',
      lastSync: now - 6 * 60000,
    },
    {
      id: 'bigquery',
      name: 'BigQuery',
      detail: 'retail_ops · marketing_events',
      status: 'connected',
      tables: 18,
      volume: '380M rows',
      lastSync: now - 22 * 60000,
    },
    {
      id: 'postgres',
      name: 'PostgreSQL',
      detail: 'billing-db · lending_core',
      status: 'connected',
      tables: 27,
      volume: '64M rows',
      lastSync: now - 3 * 60000,
    },
    { id: 's3', name: 'Amazon S3', detail: 's3://acme-raw-landing', status: 'syncing', tables: 9, volume: '2.4 TB', lastSync: now - 60 * 60000 },
    {
      id: 'salesforce',
      name: 'Salesforce',
      detail: 'CRM · Opportunities',
      status: 'error',
      tables: 6,
      volume: '4.1M rows',
      lastSync: now - 26 * 3600000,
      error: 'OAuth token expired',
    },
    { id: 'files', name: 'CSV / Excel uploads', detail: 'Local files', status: 'connected', tables: 12, volume: '12 files', lastSync: now - 2 * DAY },
  ]
}

export function seedActivity() {
  const now = Date.now()
  return [
    { id: 'a1', type: 'train', text: 'Started model search for **Loan Default Risk**: 4 candidates, 120 HPO trials', at: now - 18 * 60000 },
    { id: 'a2', type: 'improve', text: 'Critic loop improved **Weekly Store Demand Forecast**: MAPE 7.1% → 6.2%', at: now - 26 * 3600000 },
    {
      id: 'a3',
      type: 'warn',
      text: 'Input drift detected on **Card Transaction Fraud Detector** (merchant_category, PSI 0.17)',
      at: now - 30 * 3600000,
    },
    { id: 'a4', type: 'deploy', text: 'Promoted **30-Day Customer Churn v3.2** to production', at: now - 3 * DAY },
    { id: 'a5', type: 'data', text: 'Salesforce connector failed to sync: OAuth token expired', at: now - 26 * 3600000 },
  ]
}
