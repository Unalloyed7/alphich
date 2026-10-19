# THIRD_PARTY_NOTICES — initial backend increment

This starter backend uses the following Python dependencies. Versions of direct
runtime dependencies are specified in `backend/requirements.txt`; their
transitive dependency graph will need a reviewed hash-locked install before the
final hackathon submission.

| Component | Purpose | License/reference |
| --- | --- | --- |
| FastAPI | REST API / OpenAPI | MIT — https://github.com/fastapi/fastapi |
| Pydantic | Request/response validation | MIT — https://github.com/pydantic/pydantic |
| SQLAlchemy | ORM and database access | MIT — https://github.com/sqlalchemy/sqlalchemy |
| Psycopg | PostgreSQL client | LGPL-3.0 with linking exception — https://www.psycopg.org/psycopg3/docs/ |
| Uvicorn | ASGI server | BSD-3-Clause — https://github.com/encode/uvicorn |
| HTTPX | API tests | BSD-3-Clause — https://github.com/encode/httpx |
| Pytest | Unit/integration tests | MIT — https://github.com/pytest-dev/pytest |
| PostgreSQL | Persistent database | PostgreSQL License — https://www.postgresql.org/about/licence/ |
| Python | Runtime | PSF License — https://docs.python.org/3/license.html |

**External data**: the organizers' anonymized CSV datasets are NOT included and must
be downloaded separately under the hackathon's applicable terms. The illustrative
`backend/fixtures/companies.csv` is invented solely for testing.

**Bank research PDFs**: not included; several contain explicit reproduction and
distribution restrictions. Obtain permission/verify allowed use before publishing
snippets or hosting the documents outside the event's authorized environment.

**LLM models and gateways**: not integrated by this increment. Future integration
must select only a model authorized by the case and document exact model, price,
license/terms, gateway and simulated integrations here.
