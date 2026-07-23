import { useState, useCallback, useRef, useEffect } from "react";

const C = {
  bg: "#080C14",
  bgAlt: "#0C1220",
  surface: "#0F1628",
  surfaceUp: "#141E35",
  surfaceHigh: "#1A2540",
  border: "#1E2D4A",
  borderBright: "#263654",
  accent: "#3B6EFF",
  accentDim: "#1D3A8A",
  accentGlow: "#3B6EFF18",
  green: "#00D97E",
  greenDim: "#00D97E18",
  amber: "#F5A623",
  amberDim: "#F5A62318",
  orange: "#FF6B35",
  red: "#FF3D5A",
  redDim: "#FF3D5A18",
  text: "#E2EAF8",
  textSub: "#8A9ABE",
  textDim: "#3D4F72",
  purple: "#A78BFA",
  purpleDim: "#A78BFA15",
  cyan: "#22D3EE",
};

const EXAMPLE = ``;

const LANGS = [
  { v: "auto", l: "Auto-detect", icon: "◈" },
  { v: "sql", l: "SQL", icon: "◇" },
  { v: "hiveql", l: "HiveQL", icon: "⬡" },
  { v: "plsql", l: "PL/SQL", icon: "◆" },
  { v: "stored_procedure", l: "Stored Procedure", icon: "⬢" },
];

const DIALECTS = ["hive","spark","bigquery","postgres","mysql","trino"];

const RISK_MAP = {
  LOW: { color: C.green, bg: C.greenDim },
  MEDIUM: { color: C.amber, bg: C.amberDim },
  HIGH: { color: C.orange, bg: "#FF6B3518" },
  CRITICAL: { color: C.red, bg: C.redDim },
};

function useTypewriter(text, active) {
  const [displayed, setDisplayed] = useState("");
  const ref = useRef(null);
  useEffect(() => {
    if (!active || !text) { setDisplayed(text || ""); return; }
    setDisplayed("");
    let i = 0;
    const CHUNK = 4;
    ref.current = setInterval(() => {
      i += CHUNK;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) clearInterval(ref.current);
    }, 8);
    return () => clearInterval(ref.current);
  }, [text, active]);
  return displayed;
}

function Tag({ label, color, bg, small }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      padding: small ? "2px 8px" : "4px 10px",
      borderRadius: 4,
      background: bg || color + "18",
      color,
      border: `1px solid ${color}30`,
      fontSize: small ? 10 : 11,
      fontWeight: 700,
      fontFamily: "monospace",
      letterSpacing: "0.08em",
      textTransform: "uppercase",
      whiteSpace: "nowrap",
    }}>
      {label}
    </span>
  );
}

function Score({ value }) {
  const color = value >= 80 ? C.green : value >= 50 ? C.amber : C.red;
  const r = 28, stroke = 4, norm = r - stroke / 2;
  const circ = 2 * Math.PI * norm;
  const dash = (value / 100) * circ;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
      <svg width={r*2} height={r*2} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={r} cy={r} r={norm} fill="none" stroke={C.border} strokeWidth={stroke} />
        <circle cx={r} cy={r} r={norm} fill="none" stroke={color} strokeWidth={stroke}
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          style={{ transition: "stroke-dasharray 1s cubic-bezier(0.4,0,0.2,1)" }}
        />
      </svg>
      <div>
        <div style={{ fontSize: 28, fontWeight: 700, color, fontFamily: "monospace", lineHeight: 1 }}>
          {value}<span style={{ fontSize: 14, color: C.textDim, fontWeight: 400 }}>/100</span>
        </div>
        <div style={{ fontSize: 11, color: C.textSub, marginTop: 2 }}>validation score</div>
      </div>
    </div>
  );
}

function PanelHeader({ icon, title, color, right }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      marginBottom: 16, paddingBottom: 12,
      borderBottom: `1px solid ${C.border}`,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ fontSize: 14, color: color || C.textSub }}>{icon}</span>
        <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", color: color || C.textSub, textTransform: "uppercase" }}>
          {title}
        </span>
      </div>
      {right}
    </div>
  );
}

function Line({ label, value, color, mono }) {
  return (
    <div style={{
      display: "flex", justifyContent: "space-between", alignItems: "flex-start",
      padding: "7px 0", borderBottom: `1px solid ${C.border}`,
      gap: 12,
    }}>
      <span style={{ fontSize: 12, color: C.textSub, flexShrink: 0 }}>{label}</span>
      <span style={{ fontSize: 12, color: color || C.text, fontFamily: mono ? "monospace" : "inherit", textAlign: "right" }}>
        {value}
      </span>
    </div>
  );
}

function Pulse() {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span style={{
        display: "inline-block", width: 6, height: 6, borderRadius: "50%",
        background: C.accent,
        boxShadow: `0 0 0 0 ${C.accent}`,
        animation: "pulse 1.4s ease-out infinite",
      }} />
      <style>{`@keyframes pulse{0%{box-shadow:0 0 0 0 ${C.accent}88}70%{box-shadow:0 0 0 6px transparent}100%{box-shadow:0 0 0 0 transparent}}`}</style>
    </span>
  );
}

function Spinner() {
  return (
    <span style={{
      display: "inline-block", width: 14, height: 14,
      border: `2px solid ${C.accent}40`, borderTop: `2px solid ${C.accent}`,
      borderRadius: "50%", animation: "spin 0.7s linear infinite",
      marginRight: 8, verticalAlign: "middle", flexShrink: 0,
    }} />
  );
}

function GlowBtn({ onClick, disabled, loading, children }) {
  const [hover, setHover] = useState(false);
  return (
    <button onClick={onClick} disabled={disabled}
      onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}
      style={{
        flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
        gap: 8, padding: "10px 24px",
        background: disabled ? C.accentDim : hover ? "#4D7FFF" : C.accent,
        border: "none", borderRadius: 8, color: "#fff",
        fontSize: 13, fontWeight: 700, letterSpacing: "0.02em",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled && !loading ? 0.4 : 1,
        transition: "all 0.15s",
        boxShadow: hover && !disabled ? `0 0 24px ${C.accent}55` : "none",
      }}>
      {loading && <Spinner />}
      {children}
    </button>
  );
}

export default function App() {
  const [src, setSrc] = useState(EXAMPLE);
  const [lang, setLang] = useState("auto");
  const [dialect, setDialect] = useState("hive");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [animating, setAnimating] = useState(false);
  const outputText = useTypewriter(result?.pyspark_code, animating);

  const apiUrl = process.env.REACT_APP_API_URL || "http://localhost:8000";

  const migrate = useCallback(async () => {
    if (!src.trim() || loading) return;
    setLoading(true); setError(null); setResult(null); setAnimating(false);
    const t0 = Date.now();
    try {
      const res = await fetch(`${apiUrl}/migrate`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: src, source_language: lang, dialect: lang === "sql" ? dialect : undefined }),
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const data = await res.json();
      setResult({ ...data, elapsed: ((Date.now() - t0) / 1000).toFixed(1) });
      setAnimating(true);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }, [src, lang, dialect, apiUrl, loading]);

  const statusC = result
    ? result.status === "success" ? C.green
    : result.status === "low_confidence" ? C.amber : C.red
    : null;

  const riskC = result?.risk_level ? RISK_MAP[result.risk_level] || { color: C.textSub, bg: C.border } : null;

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: "'Inter','Segoe UI',system-ui,sans-serif", fontSize: 13 }}>
      <style>{`
        *{box-sizing:border-box;margin:0;padding:0}
        @keyframes spin{to{transform:rotate(360deg)}}
        @keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
        @keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
        ::selection{background:${C.accentGlow}}
        ::-webkit-scrollbar{width:4px;height:4px}
        ::-webkit-scrollbar-track{background:transparent}
        ::-webkit-scrollbar-thumb{background:${C.border};border-radius:2px}
        textarea:focus,select:focus{outline:none}
        textarea::placeholder{color:#3D4F72;font-family:'JetBrains Mono','Fira Code',monospace;font-size:12.5px;}
        .fadeIn{animation:fadeIn 0.4s ease both}
      `}</style>

      {/* Topbar */}
      <div style={{
        height: 52, background: C.bgAlt, borderBottom: `1px solid ${C.border}`,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "0 24px", position: "sticky", top: 0, zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 26, height: 26, borderRadius: 6,
            background: `linear-gradient(135deg, ${C.accent}, ${C.purple})`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 13, fontWeight: 900, color: "#fff", flexShrink: 0,
          }}>⚡</div>
          <span style={{ fontSize: 14, fontWeight: 700, letterSpacing: "-0.02em", color: C.text }}>
            Migration Copilot
          </span>
          <div style={{ width: 1, height: 16, background: C.border, margin: "0 4px" }} />
          <span style={{ fontSize: 11, color: C.textDim, fontFamily: "monospace" }}>Enterprise</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {["SQL", "HiveQL", "PL/SQL", "SP"].map(l => (
            <span key={l} style={{ fontSize: 10, color: C.textDim, fontFamily: "monospace",
              padding: "2px 6px", borderRadius: 3, border: `1px solid ${C.border}` }}>
              {l}
            </span>
          ))}
          <span style={{ fontSize: 10, color: C.textDim, margin: "0 4px" }}>→</span>
          <span style={{ fontSize: 10, color: C.accent, fontFamily: "monospace",
            padding: "2px 6px", borderRadius: 3, border: `1px solid ${C.accentDim}`,
            background: C.accentGlow }}>
            PySpark
          </span>
        </div>
      </div>

      <div style={{ padding: "20px 24px 32px", maxWidth: 1440, margin: "0 auto" }}>

        {/* Toolbar */}
        <div style={{
          display: "flex", gap: 8, marginBottom: 16, alignItems: "center",
          background: C.bgAlt, border: `1px solid ${C.border}`,
          borderRadius: 10, padding: "10px 14px",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginRight: 4 }}>
            {LANGS.map(l => (
              <button key={l.v} onClick={() => setLang(l.v)}
                style={{
                  display: "flex", alignItems: "center", gap: 5,
                  padding: "5px 12px", borderRadius: 6, border: "none",
                  background: lang === l.v ? C.accent : "transparent",
                  color: lang === l.v ? "#fff" : C.textSub,
                  fontSize: 12, fontWeight: lang === l.v ? 700 : 400,
                  cursor: "pointer", transition: "all 0.12s",
                  boxShadow: lang === l.v ? `0 0 12px ${C.accent}44` : "none",
                }}>
                <span style={{ fontSize: 10 }}>{l.icon}</span>
                {l.l}
              </button>
            ))}
          </div>

          {lang === "sql" && (
            <select value={dialect} onChange={e => setDialect(e.target.value)}
              style={{
                background: C.surface, border: `1px solid ${C.border}`,
                borderRadius: 6, color: C.textSub, padding: "5px 10px",
                fontSize: 12, cursor: "pointer",
              }}>
              {DIALECTS.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          )}

          <div style={{ flex: 1 }} />

          {result && (
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginRight: 8 }}>
              <Pulse />
              <span style={{ fontSize: 11, color: statusC, fontWeight: 600 }}>
                {result.status?.toUpperCase()}
              </span>
              <span style={{ fontSize: 11, color: C.textDim }}>{result.elapsed}s</span>
              {result.detected_language && (
                <Tag label={result.detected_language} color={C.cyan} small />
              )}
            </div>
          )}

          <GlowBtn onClick={migrate} disabled={loading || !src.trim()} loading={loading}>
            {loading ? "Migrating..." : "→ Migrate to PySpark"}
          </GlowBtn>
        </div>

        {/* Editor Row */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>

          {/* Source */}
          <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, overflow: "hidden" }}>
            <div style={{
              display: "flex", alignItems: "center", justifyContent: "space-between",
              padding: "10px 16px", borderBottom: `1px solid ${C.border}`,
              background: C.bgAlt,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ display: "flex", gap: 5 }}>
                  {["#FF5F57","#FEBC2E","#28C840"].map(c => (
                    <div key={c} style={{ width: 10, height: 10, borderRadius: "50%", background: c }} />
                  ))}
                </div>
                <span style={{ fontSize: 11, color: C.textDim, fontFamily: "monospace" }}>source.sql</span>
              </div>
              <div style={{ display: "flex", gap: 4 }}>
                {src.split("\n").length > 1 && (
                  <span style={{ fontSize: 10, color: C.textDim, fontFamily: "monospace" }}>
                    {src.split("\n").length} lines
                  </span>
                )}
              </div>
            </div>
            <div style={{ position: "relative" }}>
              <div style={{
                position: "absolute", left: 0, top: 0, bottom: 0, width: 36,
                borderRight: `1px solid ${C.border}`,
                background: C.bgAlt,
                display: "flex", flexDirection: "column",
                paddingTop: 14,
                pointerEvents: "none", userSelect: "none", zIndex: 1,
              }}>
                {src.split("\n").map((_, i) => (
                  <div key={i} style={{
                    height: "1.6em", display: "flex", alignItems: "center", justifyContent: "flex-end",
                    paddingRight: 8, fontSize: 10, color: C.textDim, fontFamily: "monospace",
                  }}>{i + 1}</div>
                ))}
              </div>
              <textarea value={src} onChange={e => setSrc(e.target.value)}
                spellCheck={false}
                placeholder="Paste your SQL, HiveQL, PL/SQL, or Stored Procedure code here...!!!!"
                style={{
                  width: "100%", minHeight: 420,
                  background: "transparent", border: "none",
                  color: C.text, fontFamily: "'JetBrains Mono','Fira Code',monospace",
                  fontSize: 12.5, lineHeight: "1.6em",
                  padding: "14px 16px 14px 52px",
                  resize: "vertical",
                }}
                onFocus={e => e.target.parentElement.parentElement.style.borderColor = C.accentDim}
                onBlur={e => e.target.parentElement.parentElement.style.borderColor = C.border}
              />
            </div>
          </div>

          {/* Output */}
          <div style={{
            background: C.surface,
            border: `1px solid ${result ? C.green + "40" : C.border}`,
            borderRadius: 10, overflow: "hidden",
            transition: "border-color 0.3s",
          }}>
            <div style={{
              display: "flex", alignItems: "center", justifyContent: "space-between",
              padding: "10px 16px", borderBottom: `1px solid ${C.border}`,
              background: C.bgAlt,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ display: "flex", gap: 5 }}>
                  {["#FF5F57","#FEBC2E","#28C840"].map(c => (
                    <div key={c} style={{ width: 10, height: 10, borderRadius: "50%", background: c }} />
                  ))}
                </div>
                <span style={{ fontSize: 11, color: C.textDim, fontFamily: "monospace" }}>output.py</span>
              </div>
              {result?.pyspark_code && (
                <button onClick={() => navigator.clipboard?.writeText(result.pyspark_code)}
                  style={{
                    background: "transparent", border: `1px solid ${C.border}`, borderRadius: 4,
                    color: C.textSub, fontSize: 10, padding: "2px 8px", cursor: "pointer",
                    fontFamily: "monospace", letterSpacing: "0.05em",
                  }}>
                  COPY
                </button>
              )}
            </div>
            <pre style={{
              minHeight: 420, margin: 0,
              padding: "14px 16px",
              fontFamily: "'JetBrains Mono','Fira Code',monospace",
              fontSize: 12.5, lineHeight: "1.6em",
              color: result?.pyspark_code ? C.text : C.textDim,
              whiteSpace: "pre-wrap", wordBreak: "break-word",
              overflow: "auto",
            }}>
              {loading
                ? <span style={{ color: C.textDim }}>
                    <span style={{ color: C.accent }}>●</span> Migrating
                    <span style={{ animation: "shimmer 1.5s infinite" }}>...</span>
                  </span>
                : outputText || "// Output appears here after migration"}
            </pre>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="fadeIn" style={{
            background: C.redDim, border: `1px solid ${C.red}30`,
            borderRadius: 8, padding: "12px 16px", marginBottom: 12,
            color: C.red, fontSize: 12, fontFamily: "monospace",
            display: "flex", alignItems: "center", gap: 8,
          }}>
            ⚠ {error}
          </div>
        )}

        {/* Result Panels */}
        {result ? (
          <div className="fadeIn" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>

            {/* Validation */}
            <div style={{
              background: C.surface, border: `1px solid ${C.border}`,
              borderRadius: 10, padding: "18px 20px",
            }}>
              <PanelHeader icon="✓" title="Validation" color={C.green}
                right={result.validation_score !== undefined &&
                  <Tag label={result.validation_score >= 80 ? "Pass" : result.validation_score >= 50 ? "Warn" : "Fail"}
                    color={result.validation_score >= 80 ? C.green : result.validation_score >= 50 ? C.amber : C.red} small />
                }
              />
              {result.validation_score !== undefined && (
                <div style={{ marginBottom: 16 }}>
                  <Score value={result.validation_score} />
                </div>
              )}
              {result.validation_issues?.length > 0 ? (
                <div>
                  {result.validation_issues.map((iss, i) => (
                    <div key={i} style={{
                      display: "flex", gap: 8, alignItems: "flex-start",
                      padding: "6px 0", borderBottom: `1px solid ${C.border}`,
                      fontSize: 12,
                    }}>
                      <span style={{ color: C.amber, flexShrink: 0, marginTop: 1 }}>▲</span>
                      <span style={{ color: C.textSub }}>{iss}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ fontSize: 12, color: C.green, display: "flex", alignItems: "center", gap: 6, marginTop: 8 }}>
                  <span>✓</span> No validation issues
                </div>
              )}
              {result.complexity && (
                <Line label="Complexity" value={result.complexity} color={C.purple} mono />
              )}
              {result.estimated_review_time && (
                <Line label="Est. review time" value={result.estimated_review_time} />
              )}
            </div>

            {/* Risk */}
            <div style={{
              background: C.surface, border: `1px solid ${riskC ? riskC.color + "30" : C.border}`,
              borderRadius: 10, padding: "18px 20px",
            }}>
              <PanelHeader icon="⚡" title="Risk Report" color={riskC?.color}
                right={result.risk_level && <Tag label={result.risk_level} color={riskC.color} bg={riskC.bg} small />}
              />

              {result.risk_score !== undefined && (
                <div style={{
                  display: "flex", alignItems: "baseline", gap: 4, marginBottom: 16,
                  padding: "12px 16px", borderRadius: 8,
                  background: riskC ? riskC.bg : C.bgAlt,
                  border: `1px solid ${riskC ? riskC.color + "20" : C.border}`,
                }}>
                  <span style={{ fontSize: 36, fontWeight: 800, color: riskC?.color || C.text, fontFamily: "monospace", lineHeight: 1 }}>
                    {result.risk_score}
                  </span>
                  <span style={{ fontSize: 14, color: C.textDim }}>/10</span>
                  <span style={{ fontSize: 11, color: C.textSub, marginLeft: 8, alignSelf: "center" }}>risk score</span>
                </div>
              )}

              {result.pii_columns?.length > 0 && (
                <div style={{ marginBottom: 14 }}>
                  <div style={{ fontSize: 10, color: C.textDim, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 }}>
                    PII Detected
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                    {result.pii_columns.map((c, i) => (
                      <Tag key={i} label={c} color={C.red} small />
                    ))}
                  </div>
                </div>
              )}

              {result.compliance_flags?.length > 0 && (
                <div>
                  <div style={{ fontSize: 10, color: C.textDim, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 }}>
                    Compliance Flags
                  </div>
                  {result.compliance_flags.map((f, i) => (
                    <div key={i} style={{ fontSize: 12, color: C.orange, padding: "5px 0", borderBottom: `1px solid ${C.border}` }}>
                      · {f}
                    </div>
                  ))}
                </div>
              )}

              {!result.pii_columns?.length && !result.compliance_flags?.length && (
                <div style={{ fontSize: 12, color: C.green, display: "flex", alignItems: "center", gap: 6 }}>
                  <span>✓</span> No compliance issues detected
                </div>
              )}
            </div>

            {/* Migration Notes */}
            <div style={{
              background: C.surface, border: `1px solid ${C.border}`,
              borderRadius: 10, padding: "18px 20px",
            }}>
              <PanelHeader icon="◈" title="Migration Notes" color={C.purple} />

              {result.procedural_flags?.length > 0 && (
                <div style={{
                  background: C.amberDim, border: `1px solid ${C.amber}20`,
                  borderRadius: 8, padding: "10px 12px", marginBottom: 14,
                }}>
                  <div style={{ fontSize: 10, color: C.amber, fontWeight: 700, letterSpacing: "0.08em", marginBottom: 6 }}>
                    ▲ MANUAL REVIEW NEEDED
                  </div>
                  {result.procedural_flags.map((f, i) => (
                    <div key={i} style={{ fontSize: 12, color: C.amber, padding: "3px 0" }}>· {f}</div>
                  ))}
                </div>
              )}

              {result.anti_patterns?.length > 0 && (
                <div style={{ marginBottom: 14 }}>
                  <div style={{ fontSize: 10, color: C.textDim, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 }}>
                    Anti-patterns
                  </div>
                  {result.anti_patterns.map((ap, i) => (
                    <div key={i} style={{ fontSize: 12, padding: "5px 0", borderBottom: `1px solid ${C.border}` }}>
                      <span style={{ color: C.red }}>✕ </span>
                      <span style={{ color: C.text }}>{typeof ap === "string" ? ap : ap.pattern}</span>
                      {ap.suggestion && (
                        <div style={{ fontSize: 11, color: C.accent, marginTop: 3, paddingLeft: 12 }}>
                          → {ap.suggestion}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {result.performance_notes?.length > 0 && (
                <div>
                  <div style={{ fontSize: 10, color: C.textDim, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 }}>
                    Performance
                  </div>
                  {result.performance_notes.map((n, i) => (
                    <div key={i} style={{ fontSize: 12, color: C.cyan, padding: "4px 0" }}>◈ {n}</div>
                  ))}
                </div>
              )}

              {!result.procedural_flags?.length && !result.anti_patterns?.length && !result.performance_notes?.length && (
                <div style={{ fontSize: 12, color: C.green, display: "flex", alignItems: "center", gap: 6 }}>
                  <span>✓</span> No migration warnings
                </div>
              )}
            </div>
          </div>
        ) : (
          /* Empty state */
          <div style={{
            display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12,
          }}>
            {[
              { icon: "✓", title: "Validation", color: C.green, desc: "Syntax check · Score · Issues" },
              { icon: "⚡", title: "Risk Report", color: C.amber, desc: "PII detection · Compliance · Risk score" },
              { icon: "◈", title: "Migration Notes", color: C.purple, desc: "Procedural flags · Anti-patterns · Performance" },
            ].map(p => (
              <div key={p.title} style={{
                background: C.surface, border: `1px solid ${C.border}`,
                borderRadius: 10, padding: "18px 20px",
                display: "flex", flexDirection: "column", minHeight: 120,
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                  <span style={{ color: p.color, fontSize: 14 }}>{p.icon}</span>
                  <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", color: p.color, textTransform: "uppercase" }}>
                    {p.title}
                  </span>
                </div>
                <div style={{ fontSize: 11, color: C.textDim, lineHeight: 1.8 }}>
                  {p.desc.split(" · ").map((s, i) => (
                    <div key={i}>· {s}</div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
