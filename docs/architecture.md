# Validrix Architecture

## Overview

Validrix is built around three architectural layers:

1. **Core** — Framework primitives (config, base classes, retry logic)
2. **Plugins** — Feature modules that hook into pytest's event system
3. **CLI** — Developer-facing commands via Click

## Plugin Lifecycle

```
pytest session start
       │
       ▼
pytest_configure()  ←── plugins register themselves here
       │
       ▼
pytest_collection_modifyitems()  ←── flaky detector wraps items
       │
       ▼
pytest_runtest_protocol()  ←── self-healing intercepts Playwright calls
       │
       ▼
pytest_sessionfinish()  ←── AI reporter summarises failures
```

## Dependency Graph

```
CLI (Click)
 └── Core (config, retry)
      └── Plugins (each independent, depends only on Core)
           ├── ai_generator  →  Claude API
           ├── ai_reporter   →  Claude API
           ├── self_healing  →  Playwright
           └── flaky_detector (pure pytest)
```

## Key Design Decisions

### Plugin Isolation
Each plugin has zero import-time dependency on other plugins.
Communication happens exclusively through pytest hooks or shared config.

### Entry Points Registration
Plugins are registered via `pyproject.toml` `[project.entry-points."pytest11"]`
so they are auto-discoverable by pytest without any user-side import.

### Pydantic for Config
Config is validated at load time with Pydantic v2 models, giving us
type safety and clear error messages before a single test runs.
