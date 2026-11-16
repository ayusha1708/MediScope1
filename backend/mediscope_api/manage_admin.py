"""List active admin emails and reset an admin password from the local CLI."""

from getpass import getpass

from pymongo.errors import PyMongoError

from mediscope_api.ai_bridge import MongoConfigurationError, create_mongo_client, get_database_name
from mediscope_api.security import hash_password, normalize_email


def main() -> None:
    try:
        with create_mongo_client() as client:
            client.admin.command("ping")
            users = client[get_database_name()]["users"]
            admins = list(
                users.find(
                    {"role": "admin", "active": True},
                    {"email": 1, "_id": 0},
                ).sort("email", 1)
            )
            if not admins:
                raise SystemExit("No active admin accounts were found in this database.")

            print("Active admin accounts:")
            for admin in admins:
                print(f"- {admin['email']}")

            try:
                email = normalize_email(input("\nAdmin email to reset: "))
            except ValueError as error:
                raise SystemExit(str(error)) from error

            admin = users.find_one({"email": email, "role": "admin", "active": True})
            if admin is None:
                raise SystemExit("No active admin account matches that email.")

            password = getpass("New admin password (at least 12 characters): ")
            confirmation = getpass("Confirm new password: ")
            if len(password) < 12:
                raise SystemExit("Password must contain at least 12 characters.")
            if password != confirmation:
                raise SystemExit("Passwords do not match.")

            users.update_one(
                {"_id": admin["_id"], "role": "admin", "active": True},
                {"$set": {"password_hash": hash_password(password)}},
            )
            print(f"Password reset for admin account: {email}")
            print("The password was not displayed or saved in project files.")
    except (MongoConfigurationError, PyMongoError) as error:
        raise SystemExit(
            "Could not access MongoDB. Resolve the Atlas connection issue before managing admins."
        ) from error


if __name__ == "__main__":
    main()
