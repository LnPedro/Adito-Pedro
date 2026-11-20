"""Testes unitários offline para as blindagens de acurácia e segurança do BI Agent (Pacote A)."""
import types
import unittest
from unittest.mock import patch

from agents.bi_agent.nodes.query_extractor import query_extractor
from agents.bi_agent.nodes.sql_validator import check_missing_null_guard
from agents.bi_agent.tool import ANSWER_SYSTEM_PROMPT
from agents.config import ORCHESTRATOR_SYSTEM_PROMPT
from core.contexts import BASE_SQL_RULES


class _FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


class PromptGuardsTests(unittest.TestCase):
    def test_answer_prompt_explicita_estimativas_e_fatos_vs_hipoteses(self) -> None:
        lower = ANSWER_SYSTEM_PROMPT.lower()
        self.assertIn("estimado", lower)
        self.assertIn("pagamento efetivamente comprovado", lower)
        self.assertIn("causas externas", lower)

    def test_orchestrator_prompt_bloqueia_vazamento_de_filtros_preservando_followup(self) -> None:
        self.assertIn("NEVER carry over filters", ORCHESTRATOR_SYSTEM_PROMPT)
        self.assertIn("E em 2025?", ORCHESTRATOR_SYSTEM_PROMPT)

    def test_base_sql_rules_proibe_where_extra_e_exige_desempate(self) -> None:
        self.assertIn("Do NOT add extra WHERE filters", BASE_SQL_RULES)
        self.assertIn("explicit tie-breaker", BASE_SQL_RULES)


class QueryExtractorDeduplicationTests(unittest.TestCase):
    def _run_extractor(self, system_prompt: str, fake_items: dict[str, list]) -> str:
        state = {
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "contexto_id": "00000000-0000-0000-0000-000000000002",
            "user_input": "quantos projetos existem?",
            "system_prompt": system_prompt,
        }
        with patch("agents.bi_agent.nodes.query_extractor.KnowledgeItem.count", return_value=3), patch(
            "agents.bi_agent.nodes.query_extractor._search_knowledge",
            side_effect=lambda _t, _c, _emb, item_type: fake_items.get(item_type, []),
        ):
            out = query_extractor(state, _FakeEmbedder())
        return out["retrieved_schema"]

    def test_remove_ddl_e_doc_quando_identicos_ao_system_prompt(self) -> None:
        ddl_text = "- vfact_projeto_pesquisa_resolved(id_projeto INTEGER)"
        doc_text = "Regras oficiais do contexto pesquisa-extensao-ifes."
        ex_text = "Pergunta: Total?\nSQL: SELECT COUNT(*) FROM vfact_projeto_pesquisa_resolved"
        fake_items = {
            "ddl": [types.SimpleNamespace(content=ddl_text)],
            "doc": [types.SimpleNamespace(content=doc_text)],
            "query_example": [types.SimpleNamespace(content=ex_text)],
        }
        retrieved = self._run_extractor(
            f"ALLOWED VIEWS:\n{ddl_text}\n\nBUSINESS RULES:\n{doc_text}",
            fake_items,
        )
        self.assertNotIn(ddl_text, retrieved)
        self.assertNotIn(doc_text, retrieved)
        self.assertIn(ex_text, retrieved)

    def test_preserva_doc_extra_nao_contido_no_system_prompt(self) -> None:
        extra_doc = "Nota adicional indexada fora do rules.md."
        fake_items = {
            "ddl": [],
            "doc": [types.SimpleNamespace(content=extra_doc)],
            "query_example": [],
        }
        retrieved = self._run_extractor("ALLOWED VIEWS:\n- v1(id INT)", fake_items)
        self.assertIn(extra_doc, retrieved)

    def test_preserva_ddl_e_doc_quando_system_prompt_vazio(self) -> None:
        ddl_text = "- vfact_projeto_pesquisa_resolved(id_projeto INTEGER)"
        fake_items = {
            "ddl": [types.SimpleNamespace(content=ddl_text)],
            "doc": [],
            "query_example": [],
        }
        retrieved = self._run_extractor("", fake_items)
        self.assertIn(ddl_text, retrieved)


class MissingNullGuardTests(unittest.TestCase):
    def test_reprova_sql_sem_is_null_em_pergunta_de_projetos_sem_coordenador(self) -> None:
        sql = (
            "SELECT COUNT(DISTINCT id_projeto) FROM vfact_projeto_pesquisa_resolved "
            "WHERE coordenador_nome IN ('Não Informado', '')"
        )
        err = check_missing_null_guard("Quantos projetos estão sem coordenador informado?", sql)
        self.assertIsNotNone(err)
        self.assertIn("coordenador_nome IS NULL", err)

    def test_reprova_sql_com_is_not_null_e_texto_vazio_sem_is_null(self) -> None:
        sql = (
            "SELECT COUNT(DISTINCT id_projeto) FROM vfact_projeto_pesquisa_resolved "
            "WHERE coordenador_nome IS NOT NULL AND coordenador_nome = 'Não Informado'"
        )
        err = check_missing_null_guard("Quantos projetos estão sem coordenador informado?", sql)
        self.assertIsNotNone(err)
        self.assertIn("coordenador_nome IS NULL", err)

    def test_reprova_sql_com_like_nao_informado_sem_is_null(self) -> None:
        sql = (
            "SELECT COUNT(DISTINCT id_projeto) FROM vfact_projeto_pesquisa_resolved "
            "WHERE coordenador_nome LIKE '%Não Informado%'"
        )
        err = check_missing_null_guard("Quantos projetos não possuem coordenador?", sql)
        self.assertIsNotNone(err)
        self.assertIn("coordenador_nome IS NULL", err)

    def test_aprova_sql_com_is_null_para_projetos_sem_coordenador(self) -> None:
        sql = (
            "SELECT COUNT(DISTINCT id_projeto) FROM vfact_projeto_pesquisa_resolved "
            "WHERE coordenador_nome IS NULL OR TRIM(coordenador_nome) = '' OR coordenador_nome = 'Não Informado'"
        )
        err = check_missing_null_guard("Quantos projetos estão sem coordenador informado?", sql)
        self.assertIsNone(err)

    def test_aprova_sql_com_coalesce(self) -> None:
        sql = (
            "SELECT COUNT(DISTINCT id_projeto) FROM vfact_projeto_pesquisa_resolved "
            "WHERE COALESCE(coordenador_nome, 'Não Informado') = 'Não Informado'"
        )
        err = check_missing_null_guard("Quantos projetos não possuem coordenador?", sql)
        self.assertIsNone(err)

    def test_nao_gera_falso_positivo_em_exclusao(self) -> None:
        sql = (
            "SELECT COUNT(DISTINCT id_projeto) FROM vfact_projeto_pesquisa_resolved "
            "WHERE coordenador_nome IS NOT NULL"
        )
        err = check_missing_null_guard("Exclua projetos sem coordenador e conte os restantes", sql)
        self.assertIsNone(err)

    def test_nao_gera_falso_positivo_em_pergunta_geral(self) -> None:
        sql = (
            "SELECT coordenador_nome, COUNT(DISTINCT id_projeto) "
            "FROM vfact_projeto_pesquisa_resolved GROUP BY coordenador_nome"
        )
        err = check_missing_null_guard("Quantos projetos cada coordenador possui?", sql)
        self.assertIsNone(err)


if __name__ == "__main__":
    unittest.main()
