from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def create_session():
    try:
        engine = create_engine('postgresql://bgardner@localhost:5432/docdb')
        Session = sessionmaker(bind=engine)
        session = Session()
    except Exception as e:
        print(f"Couldn't connect to the database: {e}")
        exit(1)
    return session
