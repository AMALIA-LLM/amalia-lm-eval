# LegalBenchPT

LegalBenchPT is a Portuguese legal reasoning benchmark covering multiple legal domains and question types. The benchmark evaluates language models' ability to understand and reason about Portuguese legal concepts, statutes, and case law.

## Dataset

**Source:** `BeatrizCanaverde/LegalBench.PT` on HuggingFace

The benchmark includes questions across 30+ legal domains, testing various aspects of legal knowledge and reasoning in Portuguese.

### Legal Domains

The benchmark covers the following areas of Portuguese law:

- **Administrative Law** - Direito Administrativo, Administrative Procedure
- **Aviation Law** - Direito da Aviação
- **Banking Law** - Direito Bancário
- **Civil Procedure** - Direito Processual Civil
- **Commercial Law** - Direito Comercial
- **Competition Law** - Direito da Concorrência
- **Constitutional Law** - Direito Constitucional
- **Contract Law** - Direito dos Contratos
- **Corporate Law** - Direito das Sociedades
- **Criminal Law** - Direito Penal, Criminal Procedure
- **Energy Law** - Direito da Energia
- **Environmental Law** - Direito Ambiental
- **EU and Community Law** - Direito Comunitário
- **Family Law** - Direito da Família
- **Financial Law** - Direito Financeiro
- **Insolvency Law** - Direito da Insolvência
- **Labor Law** - Direito do Trabalho, Labor Procedure
- **Maritime Law** - Direito Marítimo
- **Obligations Law** - Direito das Obrigações
- **Private International Law** - Direito Internacional Privado
- **Property Law** - Direito da Propriedade
- **Public Finance** - Finanças Públicas
- **Public International Law** - Direito Internacional Público
- **Securities Law** - Direito dos Valores Mobiliários
- **Succession Law** - Direito das Sucessões
- **Tax Law** - Direito Fiscal
- **Transportation Law** - Direito dos Transportes
- **Urban Planning** - Direito do Urbanismo

### Question Types

Each legal domain includes multiple question formats:

1. **Multiple Choice** - Single correct answer from multiple options
2. **Multiple Selection Questions** - Multiple correct answers to identify
3. **True/False** - Binary judgment of legal statements
4. **Cloze Tasks** - Fill-in-the-blank questions

## Task Variants

The benchmark is organized into three main variants:

### 1. Original (`original/`)
Uses standard prompting
- Tasks named: `CT_legalbench_pt_{domain}_{question_type}.yaml`

### 2. Chain-of-Thought (`cot/`)
Uses CoT prompting to elicit reasoning
- Tasks named: `CT_legalbench_pt_{domain}_{question_type}.yaml`

### 3. Log Likelihoods (`log_likelihoods/`)
Uses log-likelihood scoring for multiple choice and true/false
- Tasks named: `CT_legalbench_pt_{domain}_{question_type}.yaml`

### Task Groups

- `amalia_legalbench_pt_log` - All log-likelihood tasks
- `amalia_legalbench_pt_cot` - All chain-of-thought tasks
