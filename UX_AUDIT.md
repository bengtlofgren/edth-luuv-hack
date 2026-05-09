# UX Audit — Maritime Mission Planner UI

**File reviewed:** `/home/tyhug/hackathon/src/static/index.html`
**Date:** 2026-05-09

---

## Overall Score: 6.5 / 10

Solid foundation with good thematic consistency, but has meaningful gaps in feedback, accessibility, and handling of degraded-state scenarios.

---

## Top 5 Issues

### 1. No GPS denial state visualization (lines 87-95, 171-177)

The UI has no indicator for GPS signal quality. The "Connection Lost" overlay (line 179) only covers total WebSocket disconnection. When GPS signal degrades (partial denial, high HDOP), the vessel marker remains a confident dot/arrow, the Lat/Lon keep showing numbers, and the only hint something is wrong is the "GPS age" metric in tiny 10px text buried in the Risk panel (line 148). An operator glancing at the screen cannot tell if position data is stale.

**Severity:** High — this is a safety-critical omission for USV operations.

### 2. No position uncertainty visualization on the map (lines 213-228)

When GPS quality degrades, the vessel marker stays rendered as a precise dot (16px, no uncertainty radius). There is no uncertainty ellipse, pulsing boundary, or opacity change to indicate that the reported position is unreliable. This creates a false sense of precision for operators.

**Severity:** High — operators may make collision-critical decisions based on unreliable position data shown as precise.

### 3. No loading states for map tile layers (line 303)

When switching between base layers (Seabed/Map/Satellite), the `switchBase` function simply removes one layer and adds another with no loading spinner or placeholder. If the satellite or seabed tiles take time to load (common with WMS layers like EMODnet bathymetry at line 301), the map appears blank with no feedback to the user.

**Severity:** Medium — causes confusion during layer switching, especially on slower connections.

### 4. Risk bar gradient creates harsh color break (line 361)

The risk bar uses `linear-gradient(to right, #00d46a, #d4a000 <pct>%, #1a3050 <pct>%)` which places both color stops at the same percentage. This creates an abrupt jump from green/yellow to dark grey instead of a smooth gradient. It looks like a CSS bug rather than intentional design.

**Severity:** Low — visual polish issue, noticeable on close inspection.

### 5. No recording state indicator when sidebar is scrolled (lines 266-269)

Recording state is only visible via the Record button in the sidebar (line 121) and the elapsed time counter (line 123). If the user scrolls down in the sidebar (e.g., to look at telemetry or risk data), both indicators scroll out of view. There is no persistent recording indicator in the status bar or as a map overlay.

**Severity:** Medium — an operator could forget they are recording, impacting data collection integrity.

---

## Top 3 Suggestions for Improvement

### S1: Add GPS quality indicator badge (around lines 87-95, 175)

Insert a colored badge next to the GPS Position panel title (line 88) or in the status bar (line 175) that reflects real-time GPS fix quality: green for good, yellow for degraded, red for denied. Use the same approach as `setConnected()` (line 261) but keyed to GPS age/HDOP from the risk data (already available at line 361 via `data.imu.last_gps_sec`). This would cost ~15 lines of JS and one CSS class.

**Expected impact:** Operators can assess position confidence at a glance.

### S2: Add uncertainty circle around vessel marker (modify lines 222-228)

When heading data is null/NaN (indicating no valid GPS fix) or when GPS age exceeds a threshold, add a semi-transparent circle around the vessel marker whose radius corresponds to uncertainty. Use a Leaflet circle instead of the current crisp dot. When GPS restores, animate the circle shrinking away.

**Expected impact:** Prevents operators from making decisions based on stale position data.

### S3: Standardize animation timing variables (lines 52, 55, 216, 227)

Currently there are four different transition durations used across the UI:
- Line 52: `transition: all 0.2s` (map buttons)
- Line 55: `transition: all 0.15s` (generic buttons)
- Line 216: `transition: transform 0.25s ease` (vessel arrow)
- Line 227: `animate: true, duration: 0.3` (follow mode)

Pick one standard value (0.2s is a good middle ground) and apply it consistently. This makes the UI feel more cohesive.

**Expected impact:** Subtle but noticeable improvement in polish without any feature changes.

---

## What Works Well

1. **Waypoint placement feedback loop** — Clicking "+ WP" changes cursor to crosshair, API call shows a loading popup, success places a numbered draggable marker, failure shows a specific error message ("on land or too shallow"). This is a textbook example of good progressive feedback (lines 339-345).

2. **Dark maritime color scheme** — The `#0a1628` navy background with `#00d4aa` teal accents and `#d0d8e8` light text creates a convincing command-center aesthetic. The `backdrop-filter: blur(8px)` on the sidebar (line 16) adds a modern glassmorphism touch that fits the theme.

3. **Risk visualization on map** — Coloring waypoint segment polylines by risk score (green/yellow/red) provides an intuitive at-a-glance understanding of mission safety without needing to look at the sidebar numbers (lines 359-361).

---

## Accessibility Notes

### Color Contrast
- **Primary text (#d0d8e8) on background (#0a1628):** ~9.5:1 ratio — exceeds WCAG AAA (7:1). Good.
- **Teal accents (#00d4aa) on dark bg:** ~5.4:1 ratio — meets WCAG AA for large text but 4.5:1 is needed for normal text. Borderline acceptable.
- **Risk colors (line 62):** Green (#00d46a), yellow (#d4a000), and red (#d44040) rely solely on hue to convey severity. **Issue:** No icon, pattern, or text label accompanies the color, making it inaccessible to users with color vision deficiency (deuteranopia/protanopia). Adding a label or icon next to each risk indicator would fix this.

### Keyboard Navigation
- **Major issue:** The map cannot be navigated with a keyboard. Map pan/zoom are mouse-only.
- **Sidebar panels:** No tabindex on panel content — keyboard users cannot reach waypoint delete buttons, record button, export buttons, or replay controls without tabbing through the entire DOM.
- **Waypoint mode toggle** (line 339): Button is a `<button>` element so it's reachable, but map clicks for placing waypoints require a mouse.
- **Canvas compass** (line 199): Not accessible to screen readers — no `aria-label` or `role="img"`.

### Screen Reader Support
- No `aria-live` regions for dynamic content (GPS position updates, risk score changes, connection status).
- No `role="alert"` on the disconnect overlay (line 179) or toast notifications (line 347).
- No `aria-label` on the map container.
- Status bar changes (line 261) update text content without announcing to screen readers.

### Recommendations
1. Add `aria-live="polite"` to key update regions (GPS Position, Status, Risk Score).
2. Add `role="alert"` to the disconnect overlay and toast.
3. Add text labels beside the risk color dots (e.g., "Low", "Medium", "High") rather than relying on color alone.
4. Set focus management on the disconnect overlay when it appears (line 261).
