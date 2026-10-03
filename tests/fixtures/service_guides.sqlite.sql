CREATE TABLE service_guides (
    service_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    reviewed INTEGER NOT NULL,
    key_points_json TEXT NOT NULL,
    required_parameters_json TEXT NOT NULL,
    authentication_json TEXT NOT NULL,
    pitfalls_json TEXT NOT NULL,
    source_urls_json TEXT NOT NULL,
    PRIMARY KEY (service_id, version)
);

INSERT INTO service_guides (
    service_id, version, updated_at, reviewed, key_points_json,
    required_parameters_json, authentication_json, pitfalls_json, source_urls_json
) VALUES (
    'fixture-api', 1, '2026-10-03T00:00:00Z', 1,
    '["Use the v1 records endpoint for this fixture."]',
    '["tenant_id", "record_id"]',
    '["OAuth2 bearer token with records:read scope"]',
    '["Amounts are integer minor units.", "Follow the next_page cursor before retrying a page."]',
    '["https://example.test/fixture-api/docs", "https://example.test/fixture-api/auth"]'
);
