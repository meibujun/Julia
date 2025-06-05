import os
import tempfile
import json
import unittest

from backend.app import app, db

class SheepTestCase(unittest.TestCase):
    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp()
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + self.db_path
        app.config['TESTING'] = True
        self.client = app.test_client()
        with app.app_context():
            db.create_all()

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_create_and_get_breed(self):
        rv = self.client.post('/breeds', json={'name': 'Merino'})
        self.assertEqual(rv.status_code, 201)
        rv = self.client.get('/breeds')
        data = json.loads(rv.data)
        self.assertEqual(len(data), 1)

    def test_create_sheep(self):
        self.client.post('/breeds', json={'name': 'Suffolk'})
        rv = self.client.post('/sheep', json={'tag': '001', 'breed_id': 1, 'sex': 'M'})
        self.assertEqual(rv.status_code, 201)
        rv = self.client.get('/sheep')
        data = json.loads(rv.data)
        self.assertEqual(len(data), 1)

if __name__ == '__main__':
    unittest.main()
