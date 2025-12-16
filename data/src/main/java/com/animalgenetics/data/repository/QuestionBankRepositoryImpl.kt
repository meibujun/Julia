package com.animalgenetics.data.repository

import com.animalgenetics.domain.model.GeneticsConcept
import com.animalgenetics.domain.model.Question
import com.animalgenetics.domain.repository.QuestionBankRepository
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class QuestionBankRepositoryImpl @Inject constructor() : QuestionBankRepository {
    override suspend fun getQuestions(
        topic: GeneticsConcept,
        difficulty: Float,
        excludeIds: List<String>
    ): List<Question> {
        // Mock implementation
        return emptyList()
    }
}
