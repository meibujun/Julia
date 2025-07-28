package com.animalgenetics.domain.repository

import com.animalgenetics.domain.model.GeneticsConcept
import com.animalgenetics.domain.model.Question

interface QuestionBankRepository {
    suspend fun getQuestions(
        topic: GeneticsConcept,
        difficulty: Float,
        excludeIds: List<String>
    ): List<Question>
}
