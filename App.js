import React, { useState, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
  RadialBarChart, RadialBar, Cell, ResponsiveContainer,
} from 'recharts';

// ── Brand colours ─────────────────────────────────────────────────────────────
const C = {
  green:  '#00A651',
  navy:   '#0F2744',
  dark:   '#0A1628',
  mint:   '#00C896',
  red:    '#E53935',
  orange: '#FB8C00',
  gray:   '#64748B',
  bg:     '#F0F4F8',
};

const VERDICT_COLOR = { PASS: C.green, FAIL: C.red, WARNING: C.orange };
const VERDICT_ICON  = { PASS: '✅', FAIL: '❌', WARNING: '⚠️' };

// ── Parameter definitions ──────────────────────────────────────────────────────
const PARAM_GROUPS = {
  core: {
    label: 'Core CAD Parameters',
    params: [
      { id: 'wall_thickness_mm',     label: 'Wall Thickness',         unit: 'mm',    min: 0.5,   max: 6.0,   step: 0.1,  threshold: 2.0,   thresholdDir: 'min' },
      { id: 'fillet_radius_mm',      label: 'Fillet Radius',          unit: 'mm',    min: 0.1,   max: 5.0,   step: 0.05, threshold: 0.8,   thresholdDir: 'min' },
      { id: 'draft_angle_deg',       label: 'Draft Angle',            unit: '°',     min: 0.0,   max: 5.0,   step: 0.1,  threshold: 1.5,   thresholdDir: 'min' },
      { id: 'hole_diameter_mm',      label: 'Hole Diameter',          unit: 'mm',    min: 1.0,   max: 30.0,  step: 0.5,  threshold: null },
      { id: 'rib_height_mm',         label: 'Rib Height',             unit: 'mm',    min: 2.0,   max: 30.0,  step: 0.5,  threshold: null },
      { id: 'rib_thickness_mm',      label: 'Rib Thickness',          unit: 'mm',    min: 0.5,   max: 8.0,   step: 0.1,  threshold: null },
      { id: 'tolerance_mm',          label: 'Tolerance',              unit: 'mm',    min: 0.01,  max: 0.3,   step: 0.005, threshold: null },
      { id: 'material_density',      label: 'Material Density',       unit: 'g/cm³', min: 0.5,   max: 8.0,   step: 0.01, threshold: null },
      { id: 'surface_finish_ra',     label: 'Surface Finish Ra',      unit: 'μm',    min: 0.1,   max: 6.0,   step: 0.1,  threshold: 1.6,   thresholdDir: 'max' },
      { id: 'assembly_clearance_mm', label: 'Assembly Clearance',     unit: 'mm',    min: 0.01,  max: 1.0,   step: 0.01, threshold: [0.20, 0.50], thresholdDir: 'range' },
      { id: 'overhang_angle_deg',    label: 'Overhang Angle',         unit: '°',     min: 0.0,   max: 80.0,  step: 1.0,  threshold: 45.0,  thresholdDir: 'max' },
      { id: 'min_feature_size_mm',   label: 'Min Feature Size',       unit: 'mm',    min: 0.1,   max: 5.0,   step: 0.05, threshold: 0.6,   thresholdDir: 'min' },
      { id: 'aspect_ratio',          label: 'Aspect Ratio',           unit: ':1',    min: 1.0,   max: 15.0,  step: 0.1,  threshold: 5.0,   thresholdDir: 'max' },
      { id: 'part_weight_kg',        label: 'Part Weight',            unit: 'kg',    min: 0.01,  max: 10.0,  step: 0.05, threshold: null },
    ],
  },
  specific: {
    label: 'Product-Specific Parameters',
    params: [
      { id: 'vent_aspect_ratio',   label: 'Vent Aspect Ratio',    unit: ':1',  min: 0, max: 10.0, step: 0.1, note: 'LIGHTING only' },
      { id: 'cooling_channel_mm',  label: 'Cooling Channel Width',unit: 'mm',  min: 0, max: 15.0, step: 0.1, note: 'EV only' },
      { id: 'ip_groove_mm',        label: 'IP Sealing Groove',    unit: 'mm',  min: 0, max: 5.0,  step: 0.1, note: 'ADAS / EV' },
    ],
  },
};

const DEFAULTS = {
  product_type: 'LIGHTING',
  wall_thickness_mm: 2.5, fillet_radius_mm: 1.0, draft_angle_deg: 2.0,
  hole_diameter_mm: 8.0, rib_height_mm: 9.0, rib_thickness_mm: 3.0,
  tolerance_mm: 0.05, material_density: 1.18, surface_finish_ra: 1.2,
  assembly_clearance_mm: 0.30, overhang_angle_deg: 38.0,
  min_feature_size_mm: 1.5, aspect_ratio: 3.2, part_weight_kg: 0.45,
  vent_aspect_ratio: 3.0, cooling_channel_mm: 0.0, ip_groove_mm: 0.0,
};

// ── Utility ───────────────────────────────────────────────────────────────────
function paramOk(p, val) {
  if (!p.threshold) return null;
  if (p.thresholdDir === 'min')   return val >= p.threshold;
  if (p.thresholdDir === 'max')   return val <= p.threshold;
  if (p.thresholdDir === 'range') return val >= p.threshold[0] && val <= p.threshold[1];
  return null;
}

// ── Components ────────────────────────────────────────────────────────────────
function Badge({ verdict }) {
  const col = VERDICT_COLOR[verdict] || C.gray;
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 20,
      background: col + '22', color: col, border: `1.5px solid ${col}`,
      fontWeight: 700, fontSize: 12, letterSpacing: 1,
    }}>
      {VERDICT_ICON[verdict]} {verdict}
    </span>
  );
}

function ParamSlider({ p, value, onChange }) {
  const ok = paramOk(p, value);
  const color = ok === null ? C.gray : (ok ? C.green : C.red);
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
        <label style={{ fontSize: 12, color: '#334155', fontWeight: 600 }}>
          {p.label}
          {p.note && <span style={{ color: C.gray, fontWeight: 400, fontSize: 10, marginLeft: 4 }}>({p.note})</span>}
        </label>
        <span style={{ fontSize: 13, fontWeight: 700, color }}>
          {Number(value).toFixed(p.step < 0.1 ? 3 : p.step < 1 ? 2 : 1)} {p.unit}
        </span>
      </div>
      <input
        type="range"
        min={p.min} max={p.max} step={p.step}
        value={value}
        onChange={e => onChange(p.id, parseFloat(e.target.value))}
        style={{ width: '100%', accentColor: color, cursor: 'pointer' }}
      />
      {p.threshold !== null && (
        <div style={{ fontSize: 10, color: C.gray, marginTop: 2 }}>
          {p.thresholdDir === 'range'
            ? `Spec: ${p.threshold[0]}–${p.threshold[1]} ${p.unit}`
            : `${p.thresholdDir === 'min' ? 'Min' : 'Max'}: ${p.threshold} ${p.unit}`
          }
        </div>
      )}
    </div>
  );
}

function LayerCard({ title, badge, detail, confidence }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 10, padding: '14px 18px',
      border: '1px solid #e2e8f0', flex: 1, minWidth: 160,
    }}>
      <div style={{ fontSize: 11, color: C.gray, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>{title}</div>
      <Badge verdict={badge} />
      {confidence !== undefined && (
        <div style={{ fontSize: 11, color: C.gray, marginTop: 6 }}>Confidence: <b>{confidence}%</b></div>
      )}
      {detail && <div style={{ fontSize: 11, color: '#475569', marginTop: 6, lineHeight: 1.5 }}>{detail}</div>}
    </div>
  );
}

function ViolationItem({ v, index }) {
  const sevColor = { CRITICAL: C.red, MAJOR: C.orange, MINOR: C.mint }[v.severity] || C.gray;
  return (
    <div style={{
      background: sevColor + '10', border: `1px solid ${sevColor}44`,
      borderLeft: `4px solid ${sevColor}`, borderRadius: 6,
      padding: '10px 14px', marginBottom: 8,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontWeight: 700, fontSize: 13 }}>{v.rule_name}</span>
        <span style={{ fontSize: 11, background: sevColor + '22', color: sevColor, borderRadius: 4, padding: '2px 8px', fontWeight: 700 }}>
          {v.severity}
        </span>
      </div>
      <div style={{ fontSize: 12, color: '#475569', marginTop: 4 }}>{v.message}</div>
      <div style={{ fontSize: 11, color: C.gray, marginTop: 4 }}>
        Standard: <b>{v.standard}</b> · Context: {v.context}
      </div>
    </div>
  );
}

function ParameterChart({ form }) {
  const chartParams = [
    { key: 'wall_thickness_mm', label: 'Wall', thresh: 2.0, dir: 'min' },
    { key: 'draft_angle_deg',   label: 'Draft', thresh: 1.5, dir: 'min' },
    { key: 'fillet_radius_mm',  label: 'Fillet', thresh: 0.8, dir: 'min' },
    { key: 'assembly_clearance_mm', label: 'Clearance', thresh: 0.35, dir: 'target' },
    { key: 'aspect_ratio',      label: 'AR', thresh: 5.0, dir: 'max' },
    { key: 'surface_finish_ra', label: 'Ra', thresh: 1.6, dir: 'max' },
  ];
  const data = chartParams.map(cp => {
    const val = form[cp.key] || 0;
    const norm = cp.thresh ? val / cp.thresh : 1;
    const ok = cp.dir === 'min' ? val >= cp.thresh
              : cp.dir === 'max' ? val <= cp.thresh : true;
    return { name: cp.label, value: Math.min(norm * 100, 150), ok, actual: val, threshold: cp.thresh };
  });

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis dataKey="name" tick={{ fontSize: 11 }} />
        <YAxis tick={{ fontSize: 10 }} />
        <Tooltip formatter={(v, n, p) => [`${p.payload.actual.toFixed(2)} (${v.toFixed(0)}% of threshold)`, 'Value']} />
        <ReferenceLine y={100} stroke={C.navy} strokeDasharray="4 2" label={{ value: 'Threshold', fill: C.navy, fontSize: 10 }} />
        <Bar dataKey="value" radius={[4, 4, 0, 0]}>
          {data.map((d, i) => <Cell key={i} fill={d.ok ? C.green : C.red} fillOpacity={0.85} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [form, setForm] = useState(DEFAULTS);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('form');

  const handleChange = useCallback((id, val) => {
    setForm(f => ({ ...f, [id]: val }));
    setResult(null);
  }, []);

  const loadSample = async (pt) => {
    setForm(f => ({ ...f, product_type: pt }));
    try {
      const res = await fetch(`/sample/${pt}`);
      const data = await res.json();
      setForm(data);
      setResult(null);
    } catch {}
  };

  const runValidation = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Validation failed');
      }
      const data = await res.json();
      setResult(data);
      setActiveTab('result');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const fusion = result?.fusion;
  const rule   = result?.rule_result;
  const ml     = result?.ml_result;

  return (
    <div style={{ minHeight: '100vh', background: C.bg, fontFamily: "'Inter', system-ui, sans-serif" }}>
      {/* Header */}
      <div style={{ background: C.dark, padding: '16px 32px', display: 'flex', alignItems: 'center', gap: 16 }}>
        <div style={{ width: 36, height: 36, borderRadius: 8, background: C.green, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 18, fontWeight: 900, color: '#fff' }}>S</div>
        <div>
          <div style={{ color: '#fff', fontWeight: 800, fontSize: 18, letterSpacing: -0.5 }}>SmartCAD-AI</div>
          <div style={{ color: '#94a3b8', fontSize: 11 }}>Varroc Eureka Challenge 3.0 · PS9 · AI-Driven CAD Design Validation</div>
        </div>
        {result && (
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ color: '#94a3b8', fontSize: 12 }}>Last result:</span>
            <Badge verdict={fusion.verdict} />
          </div>
        )}
      </div>

      {/* Tab bar */}
      <div style={{ background: '#fff', borderBottom: '1px solid #e2e8f0', display: 'flex', gap: 0, padding: '0 24px' }}>
        {[
          { id: 'form', label: '⚙️ Design Parameters' },
          { id: 'result', label: '📊 Validation Report', disabled: !result },
        ].map(t => (
          <button key={t.id} onClick={() => !t.disabled && setActiveTab(t.id)}
            style={{
              border: 'none', background: 'none', padding: '12px 20px',
              fontWeight: 600, fontSize: 13, cursor: t.disabled ? 'default' : 'pointer',
              color: activeTab === t.id ? C.navy : t.disabled ? '#cbd5e1' : C.gray,
              borderBottom: activeTab === t.id ? `2px solid ${C.navy}` : '2px solid transparent',
              transition: 'all .15s',
            }}>
            {t.label}
          </button>
        ))}
      </div>

      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 20px' }}>

        {/* ── FORM TAB ─────────────────────────────────────────────────────── */}
        {activeTab === 'form' && (
          <div style={{ display: 'grid', gridTemplateColumns: '340px 1fr', gap: 20, alignItems: 'start' }}>

            {/* Left panel: product type + chart */}
            <div>
              <div style={{ background: '#fff', borderRadius: 12, padding: 20, border: '1px solid #e2e8f0', marginBottom: 16 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: C.gray, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>Product Family</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  {['LIGHTING','EV','ADAS','STRUCTURAL'].map(pt => (
                    <button key={pt} onClick={() => loadSample(pt)}
                      style={{
                        padding: '10px 8px', borderRadius: 8, border: `2px solid`,
                        borderColor: form.product_type === pt ? C.navy : '#e2e8f0',
                        background: form.product_type === pt ? C.navy : '#f8fafc',
                        color: form.product_type === pt ? '#fff' : '#334155',
                        fontWeight: 700, fontSize: 12, cursor: 'pointer', transition: 'all .15s',
                      }}>
                      {{'LIGHTING':'💡 Lighting','EV':'⚡ EV','ADAS':'📡 ADAS','STRUCTURAL':'⚙️ Structural'}[pt]}
                    </button>
                  ))}
                </div>
                <div style={{ fontSize: 11, color: C.gray, marginTop: 8 }}>Click to load sample design</div>
              </div>

              <div style={{ background: '#fff', borderRadius: 12, padding: 20, border: '1px solid #e2e8f0', marginBottom: 16 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: C.gray, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>Parameter Health</div>
                <ParameterChart form={form} />
                <div style={{ fontSize: 10, color: C.gray, textAlign: 'center', marginTop: 4 }}>% of threshold | above 100 = rule met</div>
              </div>

              <button onClick={runValidation} disabled={loading}
                style={{
                  width: '100%', padding: '14px', borderRadius: 10, border: 'none',
                  background: loading ? C.gray : C.green, color: '#fff',
                  fontWeight: 800, fontSize: 15, cursor: loading ? 'not-allowed' : 'pointer',
                  boxShadow: loading ? 'none' : '0 4px 14px #00A65140',
                  transition: 'all .2s', letterSpacing: 0.3,
                }}>
                {loading ? '⏳ Validating…' : '▶ Run Validation'}
              </button>
              {error && <div style={{ marginTop: 10, color: C.red, fontSize: 12, padding: '8px 12px', background: '#fef2f2', borderRadius: 6 }}>⚠️ {error}</div>}
            </div>

            {/* Right panel: parameter sliders */}
            <div>
              {Object.entries(PARAM_GROUPS).map(([gid, group]) => (
                <div key={gid} style={{ background: '#fff', borderRadius: 12, padding: 20, border: '1px solid #e2e8f0', marginBottom: 16 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: C.gray, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 16 }}>{group.label}</div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '0 28px' }}>
                    {group.params.map(p => (
                      <ParamSlider key={p.id} p={p} value={form[p.id] ?? 0} onChange={handleChange} />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── RESULT TAB ───────────────────────────────────────────────────── */}
        {activeTab === 'result' && result && (
          <div>
            {/* Hero verdict */}
            <div style={{
              background: (VERDICT_COLOR[fusion.verdict] || C.gray) + '12',
              border: `2px solid ${VERDICT_COLOR[fusion.verdict] || C.gray}`,
              borderRadius: 14, padding: '24px 28px', marginBottom: 20,
              display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap',
            }}>
              <div style={{ fontSize: 56 }}>{VERDICT_ICON[fusion.verdict]}</div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 28, fontWeight: 900, color: VERDICT_COLOR[fusion.verdict], letterSpacing: -1 }}>
                  {fusion.verdict}
                </div>
                <div style={{ fontSize: 14, color: '#475569', marginTop: 4 }}>{fusion.method}</div>
                <div style={{ fontSize: 13, color: C.gray, marginTop: 4 }}>
                  Fusion confidence: <b>{fusion.confidence}%</b> · Product: <b>{result.product_type}</b>
                </div>
              </div>
              {fusion.verdict === 'FAIL' && (
                <div style={{ background: C.red + '15', border: `1px solid ${C.red}40`, borderRadius: 8, padding: '10px 14px', fontSize: 12, color: C.red, fontWeight: 600 }}>
                  ⚠️ Do not proceed to tooling.<br/>Rework required.
                </div>
              )}
              {fusion.verdict === 'WARNING' && (
                <div style={{ background: C.orange + '15', border: `1px solid ${C.orange}40`, borderRadius: 8, padding: '10px 14px', fontSize: 12, color: C.orange, fontWeight: 600 }}>
                  🔍 Escalate to Design Lead<br/>for manual review.
                </div>
              )}
            </div>

            {/* 3-Layer summary */}
            <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
              <LayerCard title="Layer 1 — Rule Engine" badge={rule.rule_verdict}
                detail={`${rule.violations.length} violation(s) · ${rule.critical_count} CRITICAL · ${rule.major_count} MAJOR · ${rule.minor_count} MINOR`} />
              <LayerCard title="Layer 2 — GBM Model" badge={ml.ml_verdict}
                confidence={ml.confidence}
                detail={`Pass prob: ${ml.pass_prob}% · Fail prob: ${ml.fail_prob}%`} />
              <LayerCard title="Layer 3 — Fusion" badge={fusion.verdict}
                confidence={fusion.confidence}
                detail={fusion.method} />
            </div>

            {/* ML probability bar */}
            <div style={{ background: '#fff', borderRadius: 12, padding: 20, border: '1px solid #e2e8f0', marginBottom: 20 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: C.gray, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 14 }}>GBM Probability Distribution</div>
              <div style={{ marginBottom: 10 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                  <span style={{ color: C.green, fontWeight: 700 }}>PASS</span>
                  <span style={{ fontWeight: 700 }}>{ml.pass_prob}%</span>
                </div>
                <div style={{ height: 16, background: '#f1f5f9', borderRadius: 8, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${ml.pass_prob}%`, background: C.green, borderRadius: 8, transition: 'width .5s' }} />
                </div>
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                  <span style={{ color: C.red, fontWeight: 700 }}>FAIL</span>
                  <span style={{ fontWeight: 700 }}>{ml.fail_prob}%</span>
                </div>
                <div style={{ height: 16, background: '#f1f5f9', borderRadius: 8, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${ml.fail_prob}%`, background: C.red, borderRadius: 8, transition: 'width .5s' }} />
                </div>
              </div>
            </div>

            {/* Violations */}
            {rule.violations.length > 0 ? (
              <div style={{ background: '#fff', borderRadius: 12, padding: 20, border: '1px solid #e2e8f0', marginBottom: 20 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: C.gray, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 14 }}>
                  Rule Violations ({rule.violations.length})
                </div>
                {rule.violations.map((v, i) => <ViolationItem key={i} v={v} index={i} />)}
              </div>
            ) : (
              <div style={{ background: C.green + '10', border: `1px solid ${C.green}40`, borderRadius: 12, padding: 18, marginBottom: 20, color: C.green, fontWeight: 700 }}>
                ✅ No rule violations detected
              </div>
            )}

            {/* Passed rules */}
            {rule.passed_rules.length > 0 && (
              <div style={{ background: '#fff', borderRadius: 12, padding: 20, border: '1px solid #e2e8f0', marginBottom: 20 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: C.gray, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 14 }}>
                  Passed Rules ({rule.passed_rules.length})
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  {rule.passed_rules.map((r, i) => (
                    <span key={i} style={{ fontSize: 12, background: C.green + '15', color: C.green, borderRadius: 6, padding: '4px 10px', fontWeight: 600 }}>
                      ✓ {r.rule_name}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <button onClick={() => setActiveTab('form')}
              style={{ padding: '10px 24px', borderRadius: 8, border: `2px solid ${C.navy}`, background: 'none', color: C.navy, fontWeight: 700, cursor: 'pointer', fontSize: 13 }}>
              ← Edit Parameters
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
