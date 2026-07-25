from web import create_app


app = create_app()


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
