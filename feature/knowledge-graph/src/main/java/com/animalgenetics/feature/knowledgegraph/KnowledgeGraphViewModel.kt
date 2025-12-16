package com.animalgenetics.feature.knowledgegraph

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.animalgenetics.domain.model.GeneticsConcept
import com.animalgenetics.domain.usecase.GetGraphConceptsUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class KnowledgeGraphViewModel @Inject constructor(
    private val getGraphConceptsUseCase: GetGraphConceptsUseCase
) : ViewModel() {

    private val _uiState = MutableStateFlow<KnowledgeGraphUiState>(KnowledgeGraphUiState.Loading)
    val uiState: StateFlow<KnowledgeGraphUiState> = _uiState

    fun loadConcepts() {
        viewModelScope.launch {
            _uiState.value = KnowledgeGraphUiState.Loading
            getGraphConceptsUseCase.execute()
                .onSuccess { concepts ->
                    _uiState.value = KnowledgeGraphUiState.Success(concepts)
                }
                .onFailure { error ->
                    _uiState.value = KnowledgeGraphUiState.Error(error.message ?: "Unknown error")
                }
        }
    }
}

sealed class KnowledgeGraphUiState {
    object Loading : KnowledgeGraphUiState()
    data class Success(val concepts: List<GeneticsConcept>) : KnowledgeGraphUiState()
    data class Error(val message: String) : KnowledgeGraphUiState()
}
