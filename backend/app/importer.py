"""Run with: python -m app.importer /data/raw/df_Uid_data_hack.csv [--replace]."""
import argparse
import json

from app.core import Base, import_companies, make_session_factory


def main():
    parser = argparse.ArgumentParser(description="Load the anonymized company CSV")
    parser.add_argument("csv_path")
    parser.add_argument("--replace", action="store_true", help="Replace all profiles and invalidate demo sessions")
    options = parser.parse_args()
    engine, sessions = make_session_factory()
    Base.metadata.create_all(engine)
    print(json.dumps(import_companies(options.csv_path, sessions, replace=options.replace), ensure_ascii=False))


if __name__ == "__main__":
    main()
