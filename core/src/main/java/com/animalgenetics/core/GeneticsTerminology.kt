package com.animalgenetics.core

import com.animalgenetics.domain.model.TermTranslation

object GeneticsTerminology {
    private val terms = mapOf(
        "dna_replication" to TermTranslation(
            en = "DNA Replication",
            zh_CN = "DNA复制",
            zh_TW = "DNA複製"
        ),
        "gene_expression" to TermTranslation(
            en = "Gene Expression",
            zh_CN = "基因表达",
            zh_TW = "基因表現"
        ),
        "mendelian_inheritance" to TermTranslation(
            en = "Mendelian Inheritance",
            zh_CN = "孟德尔遗传",
            zh_TW = "孟德爾遺傳"
        )
    )

    fun getEnglishTerm(key: String): String = terms[key]?.en ?: key
    fun getChineseTerm(key: String): String = terms[key]?.zh_CN ?: key
}
