package com.animalgenetics.feature.virtuallab

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import com.animalgenetics.domain.model.ExperimentAction
import com.animalgenetics.domain.model.ExperimentState
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject

@HiltViewModel
class VirtualLabViewModel @Inject constructor() : ViewModel() {

    private val _experimentState = MutableLiveData<ExperimentState>()
    val experimentState: LiveData<ExperimentState> = _experimentState

    fun loadExperiment(experimentType: String?) {
        if (experimentType == null) {
            _experimentState.value = ExperimentState.Error("Experiment type not specified.")
            return
        }
        // In a real app, you would fetch the config for the experiment type
        val config = com.animalgenetics.domain.model.ExperimentConfig(experimentType, emptyMap())
        _experimentState.value = ExperimentState.Ready(config)
    }

    fun processExperimentAction(action: ExperimentAction) {
        // Process actions received from Unity
        // e.g., update state, calculate results, etc.
    }
}
