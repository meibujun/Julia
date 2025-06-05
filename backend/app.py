from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'sheep.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
ma = Marshmallow(app)

class Breed(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), unique=True, nullable=False)

class BreedSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Breed
        load_instance = True

class Sheep(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tag = db.Column(db.String(64), unique=True, nullable=False)
    breed_id = db.Column(db.Integer, db.ForeignKey('breed.id'))
    birth_date = db.Column(db.Date)
    sex = db.Column(db.String(1))

    breed = db.relationship('Breed')

class SheepSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Sheep
        include_fk = True
        load_instance = True

class BreedingRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sire_id = db.Column(db.Integer, db.ForeignKey('sheep.id'))
    dam_id = db.Column(db.Integer, db.ForeignKey('sheep.id'))
    offspring_id = db.Column(db.Integer, db.ForeignKey('sheep.id'))
    date = db.Column(db.Date)

    sire = db.relationship('Sheep', foreign_keys=[sire_id])
    dam = db.relationship('Sheep', foreign_keys=[dam_id])
    offspring = db.relationship('Sheep', foreign_keys=[offspring_id])

class BreedingRecordSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = BreedingRecord
        include_fk = True
        load_instance = True

breed_schema = BreedSchema()
breeds_schema = BreedSchema(many=True)
sheep_schema = SheepSchema()
sheeps_schema = SheepSchema(many=True)
record_schema = BreedingRecordSchema()
records_schema = BreedingRecordSchema(many=True)

@app.before_first_request
def create_tables():
    db.create_all()

# Breed endpoints
@app.route('/breeds', methods=['POST'])
def create_breed():
    name = request.json.get('name')
    breed = Breed(name=name)
    db.session.add(breed)
    db.session.commit()
    return breed_schema.jsonify(breed), 201

@app.route('/breeds', methods=['GET'])
def get_breeds():
    return breeds_schema.jsonify(Breed.query.all())

# Sheep endpoints
@app.route('/sheep', methods=['POST'])
def create_sheep():
    data = request.get_json()
    sheep = Sheep(**data)
    db.session.add(sheep)
    db.session.commit()
    return sheep_schema.jsonify(sheep), 201

@app.route('/sheep', methods=['GET'])
def get_sheep():
    return sheeps_schema.jsonify(Sheep.query.all())

# Breeding record endpoints
@app.route('/records', methods=['POST'])
def create_record():
    data = request.get_json()
    record = BreedingRecord(**data)
    db.session.add(record)
    db.session.commit()
    return record_schema.jsonify(record), 201

@app.route('/records', methods=['GET'])
def get_records():
    return records_schema.jsonify(BreedingRecord.query.all())

if __name__ == '__main__':
    app.run(debug=True)
