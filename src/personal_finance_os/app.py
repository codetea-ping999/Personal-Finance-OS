from __future__ import annotations

import uvicorn


def run() -> None:
    uvicorn.run("personal_finance_os.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
