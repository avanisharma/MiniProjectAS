-- Run with: sqlite3 data/rfp_evaluation.db < seed.sql
CREATE TABLE IF NOT EXISTS evaluation_criteria (
    criterion_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    weight REAL NOT NULL CHECK(weight >= 0),
    max_score REAL NOT NULL CHECK(max_score > 0),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0,1))
);
INSERT OR IGNORE INTO evaluation_criteria VALUES
(1, 'Technical Capability', 'Architecture, integrations, scalability, and technical fit.', 30, 10, 1),
(2, 'Implementation Plan', 'Timeline, milestones, staffing, and delivery risk plan.', 20, 10, 1),
(3, 'Commercial Value', 'Pricing clarity, total cost, and commercial assumptions.', 20, 10, 1),
(4, 'Security & Compliance', 'Security controls, certifications, privacy, and auditability.', 20, 10, 1),
(5, 'Support & Experience', 'Support model, similar projects, and references.', 10, 10, 1);
