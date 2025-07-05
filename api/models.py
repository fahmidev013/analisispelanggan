from sqlalchemy import Column, Integer, Float, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Company(Base):
    __tablename__ = "companies"
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    address = Column(Text)
    phone = Column(Integer)
    web = Column(Text)
    rating = Column(Text)
    user_ratings_count = Column(Integer)
    reviews = Column(Text)
    types = Column(Text)
    profile_info = Column(Text)

def get_engine_and_session(db_url):
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session()
