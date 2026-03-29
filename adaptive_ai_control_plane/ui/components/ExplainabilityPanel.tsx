import React from "react";

/** Flight-recorder metadata for the most recent routing decision (Nexus-Q). */
export type ExplainabilityPayload = {
  stateVector: {
    promptComplexity: string;
    providerHealth: Record<string, string>;
    latencyHistory: Record<string, number[]>;
  };
  blockedProviders: string[];
  qValues: Record<string, number>;
  quantumRandomizationFired: boolean;
};

type Props = {
  data: ExplainabilityPayload | null;
  className?: string;
};

const barWidth = (score: number, max: number) =>
  max > 0 ? `${Math.max(6, (Math.abs(score) / max) * 100)}%` : "6%";

/**
 * “Glass box” explainability — flight data recorder styling for judge demos.
 */
export function ExplainabilityPanel({ data, className }: Props) {
  if (!data) {
    return (
      <aside
        className={className}
        style={styles.shell}
        aria-label="Explainability panel"
      >
        <header style={styles.header}>
          <span style={styles.recDot} />
          <span style={styles.title}>EXPLAINABILITY / FDR</span>
          <span style={styles.muted}>Awaiting routing decision…</span>
        </header>
      </aside>
    );
  }

  const { stateVector, blockedProviders, qValues, quantumRandomizationFired } =
    data;
  const qEntries = Object.entries(qValues);
  const maxQ = Math.max(1e-6, ...qEntries.map(([, v]) => Math.abs(v)));

  return (
    <aside
      className={className}
      style={styles.shell}
      aria-label="Explainability panel"
    >
      <header style={styles.header}>
        <span style={styles.recDot} />
        <span style={styles.title}>EXPLAINABILITY / FDR</span>
        <span
          style={{
            ...styles.badge,
            ...(quantumRandomizationFired ? styles.badgeOn : styles.badgeOff),
          }}
        >
          Quantum randomization fired
        </span>
      </header>

      <section style={styles.section}>
        <h3 style={styles.h3}>State vector</h3>
        <dl style={styles.dl}>
          <dt style={styles.dt}>Prompt complexity</dt>
          <dd style={styles.dd}>{stateVector.promptComplexity}</dd>
          <dt style={styles.dt}>Provider health</dt>
          <dd style={styles.dd}>
            <ul style={styles.ul}>
              {Object.entries(stateVector.providerHealth).map(([k, v]) => (
                <li key={k}>
                  <code style={styles.code}>{k}</code> — {v}
                </li>
              ))}
            </ul>
          </dd>
          <dt style={styles.dt}>Latency history (recent)</dt>
          <dd style={styles.dd}>
            <ul style={styles.ul}>
              {Object.entries(stateVector.latencyHistory).map(([k, arr]) => (
                <li key={k}>
                  <code style={styles.code}>{k}</code> —{" "}
                  {arr.length ? arr.map((x) => x.toFixed(3)).join(", ") : "—"}
                </li>
              ))}
            </ul>
          </dd>
        </dl>
      </section>

      <section style={styles.section}>
        <h3 style={styles.h3}>Blocked providers (SLA)</h3>
        {blockedProviders.length ? (
          <ul style={styles.ul}>
            {blockedProviders.map((p) => (
              <li key={p}>
                <span style={styles.blocked}>{p}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p style={styles.muted}>None</p>
        )}
      </section>

      <section style={styles.section}>
        <h3 style={styles.h3}>Q-values (RL scores)</h3>
        <div style={styles.qGrid}>
          {qEntries.map(([name, score]) => (
            <div key={name} style={styles.qRow}>
              <span style={styles.qName}>{name}</span>
              <div style={styles.qTrack}>
                <div
                  style={{
                    ...styles.qBar,
                    width: barWidth(score, maxQ),
                  }}
                  title={`${score.toFixed(4)}`}
                />
              </div>
              <span style={styles.qNum}>{score.toFixed(3)}</span>
            </div>
          ))}
        </div>
      </section>
    </aside>
  );
}

const styles: Record<string, React.CSSProperties> = {
  shell: {
    fontFamily: '"JetBrains Mono", "Consolas", monospace',
    background: "linear-gradient(165deg, rgba(12,14,22,0.96), rgba(8,9,14,0.98))",
    border: "1px solid rgba(0,245,212,0.22)",
    borderRadius: 10,
    padding: "14px 16px",
    color: "#e8f0ff",
    boxShadow: "0 12px 40px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06)",
  },
  header: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    marginBottom: 12,
    borderBottom: "1px solid rgba(0,245,212,0.15)",
    paddingBottom: 10,
  },
  recDot: {
    width: 10,
    height: 10,
    borderRadius: "50%",
    background: "#ff4d6d",
    boxShadow: "0 0 12px rgba(255,77,109,0.8)",
  },
  title: {
    letterSpacing: "0.12em",
    fontSize: 11,
    fontWeight: 600,
    color: "#9fe8df",
  },
  muted: { fontSize: 12, opacity: 0.65 },
  badge: {
    marginLeft: "auto",
    fontSize: 10,
    letterSpacing: "0.06em",
    padding: "4px 10px",
    borderRadius: 999,
    border: "1px solid rgba(255,255,255,0.12)",
    textTransform: "uppercase",
  },
  badgeOn: {
    color: "#0a1620",
    background: "linear-gradient(90deg, #00f5d4, #7b61ff)",
    border: "none",
    fontWeight: 700,
  },
  badgeOff: {
    color: "rgba(255,255,255,0.45)",
    background: "rgba(255,255,255,0.04)",
  },
  section: { marginBottom: 14 },
  h3: {
    margin: "0 0 8px 0",
    fontSize: 11,
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    color: "rgba(0,245,212,0.85)",
  },
  dl: { margin: 0 },
  dt: { fontSize: 10, opacity: 0.55, marginTop: 6 },
  dd: { margin: "2px 0 0 0", fontSize: 12 },
  ul: { margin: "4px 0 0 16px", padding: 0 },
  code: { color: "#b8c7ff" },
  blocked: {
    color: "#ffb4b4",
    fontWeight: 600,
  },
  qGrid: { display: "flex", flexDirection: "column", gap: 6 },
  qRow: {
    display: "grid",
    gridTemplateColumns: "120px 1fr 52px",
    alignItems: "center",
    gap: 8,
  },
  qName: { fontSize: 11, opacity: 0.9, overflow: "hidden", textOverflow: "ellipsis" },
  qTrack: {
    height: 8,
    background: "rgba(255,255,255,0.06)",
    borderRadius: 4,
    overflow: "hidden",
  },
  qBar: {
    height: "100%",
    background: "linear-gradient(90deg, #00f5d4, #7b61ff)",
    borderRadius: 4,
  },
  qNum: { fontSize: 11, textAlign: "right", opacity: 0.85 },
};

export default ExplainabilityPanel;
