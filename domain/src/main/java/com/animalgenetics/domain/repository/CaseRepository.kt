package com.animalgenetics.domain.repository

import com.animalgenetics.domain.model.CaseStudy

interface CaseRepository {
    suspend fun getCase(caseId: String): CaseStudy
}
