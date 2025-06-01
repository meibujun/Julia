import unittest
from genostockpy.core.model_parser import parse_model_equation_py, ParsedTerm

class TestModelParser(unittest.TestCase):

    def test_parse_simple_st_fixed_effects(self):
        eq = "y ~ mu + age + sex"
        data_cols = ["age", "sex", "y"] # age=covariate, sex=factor
        cov_cols = ["age"]

        responses, terms_list, terms_map = parse_model_equation_py(eq, data_cols, covariate_columns_user=cov_cols)

        self.assertListEqual(responses, ["y"])
        self.assertEqual(len(terms_list), 3) # mu (intercept), age, sex
        self.assertEqual(len(terms_map), 3)

        term_mu = terms_map.get("y:intercept") # Assuming mu maps to intercept
        self.assertIsNotNone(term_mu)
        self.assertEqual(term_mu.term_type, "intercept")
        self.assertEqual(term_mu.base_name, "intercept")

        term_age = terms_map.get("y:age")
        self.assertIsNotNone(term_age)
        self.assertEqual(term_age.term_type, "covariate")
        self.assertTrue(term_age.is_covariate)
        self.assertTrue(term_age.is_fixed)

        term_sex = terms_map.get("y:sex")
        self.assertIsNotNone(term_sex)
        self.assertEqual(term_sex.term_type, "factor") # Identified as factor because in data_cols but not cov_cols
        self.assertFalse(term_sex.is_covariate)
        self.assertTrue(term_sex.is_fixed)

    def test_parse_st_with_random_effect(self):
        eq = "yield ~ 1 + fixed_cov + animal_id" # 1 is intercept
        data_cols = ["yield", "fixed_cov"]
        cov_cols = ["fixed_cov"]
        rand_effects = ["animal_id"]

        responses, _, terms_map = parse_model_equation_py(eq, data_cols,
                                                          random_effects_user=rand_effects,
                                                          covariate_columns_user=cov_cols)
        self.assertListEqual(responses, ["yield"])
        self.assertIn("yield:intercept", terms_map)
        self.assertIn("yield:fixed_cov", terms_map)
        self.assertIn("yield:animal_id", terms_map)

        term_animal = terms_map["yield:animal_id"]
        self.assertTrue(term_animal.is_random)
        self.assertFalse(term_animal.is_fixed)
        # term_type could be 'random_factor' or 'unknown_potential_random' if not in data_cols
        self.assertIn(term_animal.term_type, ["random_factor", "unknown_potential_random"])


    def test_parse_multi_trait_model(self):
        eq = "T1 ~ mu + age; T2 ~ mu + Parity"
        data_cols = ["T1", "T2", "age", "Parity"]
        cov_cols = ["age"] # Parity will be factor

        responses, terms_list, terms_map = parse_model_equation_py(eq, data_cols, covariate_columns_user=cov_cols)

        self.assertListEqual(responses, ["T1", "T2"])
        self.assertEqual(len(terms_map), 4) # T1:intercept, T1:age, T2:intercept, T2:Parity

        self.assertIn("T1:intercept", terms_map)
        self.assertIn("T1:age", terms_map)
        self.assertEqual(terms_map["T1:age"].term_type, "covariate")

        self.assertIn("T2:intercept", terms_map)
        self.assertIn("T2:Parity", terms_map)
        self.assertEqual(terms_map["T2:Parity"].term_type, "factor")

    def test_parse_interaction_term(self):
        eq = "y ~ A + B + A*B"
        data_cols = ["y", "A", "B"] # Assume A, B are factors

        responses, _, terms_map = parse_model_equation_py(eq, data_cols)

        self.assertListEqual(responses, ["y"])
        self.assertIn("y:A", terms_map)
        self.assertIn("y:B", terms_map)
        self.assertIn("y:A*B", terms_map)

        term_interaction = terms_map["y:A*B"]
        self.assertEqual(term_interaction.term_type, "interaction")
        self.assertListEqual(term_interaction.factors, ["A", "B"])

    def test_parse_empty_rhs(self):
        eq = "y ~ 1" # Intercept only
        data_cols = ["y"]
        responses, _, terms_map = parse_model_equation_py(eq, data_cols)
        self.assertListEqual(responses, ["y"])
        self.assertEqual(len(terms_map), 1)
        self.assertIn("y:intercept", terms_map)

    def test_invalid_equation(self):
        with self.assertRaises(ValueError):
            parse_model_equation_py("y  mu + age", ["y", "age"]) # Missing separator
        with self.assertRaises(ValueError):
            parse_model_equation_py("~ mu + age", ["age"]) # Missing response

if __name__ == '__main__':
    unittest.main()
```

**Phase 2: Unit Tests for MME Builder (`genostockpy/tests/core/test_mme_builder.py`)**
