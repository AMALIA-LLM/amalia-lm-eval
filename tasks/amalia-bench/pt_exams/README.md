# PHEB: An European Portuguese High School-Level LLM Benchmark

[![Paper](https://img.shields.io/badge/Paper-LREC-blue)](https://lrec.elra.info/lrec2026-main-367)
[![Original Repo](https://img.shields.io/badge/Repo-PHEB-green)](https://github.com/AMALIA-LLM/pheb)

## Task Overview
This task evaluates language models on Portuguese high school-level exam questions through multiple-choice question answering. The benchmark comprises 1,819 questions across six subjects: Geography (424 questions), Philosophy (373), Biology and Geology (332), Portuguese (306), Mathematics A (264), and History A (120). Questions typically include supporting documents (text excerpts, diagrams, or data) followed by a question stem and four answer options. Models must demonstrate reading comprehension, domain knowledge, and reasoning skills to select the correct answer.

**Paper**: https://lrec.elra.info/lrec2026-main-367

The benchmark includes four task variations:
- **default**: Perplexity-based evaluation (log-likelihood scoring)
- **default_chat**: Chat-formatted prompts with perplexity scoring
- **cot**: Chain-of-thought prompting with answer extraction via regex

## Dataset
- **Name** : `pt_exams`
- **Dataset Source**: [pt_exams](https://huggingface.co/datasets/amalia-llm/pt_exams)
- **Description**: This dataset contains multiple-choice questions from Portuguese high school exams. Each question generally consists of a supporting document and question with 4 answer options.


Each dataset instance consists of:
- **year**: The year of the exam.
- **phase**: The phase of the exam (e.g., 1st phase, 2nd phase).
- **subject**: The subject of the exam (e.g., Portuguese, Mathematics).
- **question_group**: The group of questions (e.g., GRUPO II).
- **question_number**: The number of the question.
- **question**: The text of the question, which may include a supporting document.
- **answer**: The index of the correct answer (0-based).
- **choices**: A list of possible answer choices.
- **is_completion**: A boolean indicating if the task is MCQ of completion. E.g. fill in the blanks question.


## Task Definition
The task is to select the correct answer from the provided choices based on the context of the question. The model should be able to understand the question and the context, and then choose the most appropriate answer from the given options.

### Example Prompt
```
Documento document_2:
Não creio que as classificações acrescentem o que quer que seja à fruição de uma obra de arte, seja ela literária ou de outra qualquer natureza; mas também não penso que a prejudiquem. Por isso, ao terminar a leitura de Memorial do Convento, ainda sob o império da fascinação que ela me provocou, surpreendi-me a interrogar: que livro é este? Que escreveu, que quis José Saramago escrever? Um romance histórico? Um romance realista? Uma alegoria? Uma parábola? Uma epopeia? Um conto de fadas? Uma história de amor? [...] A resposta surgiu, inevitável, irrecusável por assim dizer: Memorial do Convento é tudo isso, um caleidoscópio, [...] um espelho do real reinventado, diverso e complexo, à imagem e semelhança do mundo e dos homens que nele habitam e o fazem avançar. [...]Este romance não é apenas um romance histórico, a sua duração transcende os limites cronológicos do quadro histórico em que à primeira vista parece encerrar-se [...]. A cada passo surgem referências a acontecimentos que virão a produzir-se muito para além do marco temporal da história que nos é contada a propósito dos damascos carmesins e dos panos verdes que ornamentam o coro da Igreja alude-se ao «gosto português pelo verde e pelo encarnado, que, em vindo uma república, dará bandeira» [...]. É que, neste romance[...], a história não é uma categoria imutável e fixa, mas a contínua respiração da realidade, rio cujas águas nunca param e nunca se repetem. [...]
A luta de classes. De outra coisa não fala este romance que por isso mesmo é também um romance realista e uma epopeia. Às duas epígrafes¹ que o autor lhe antepôs uma do Padre Manuel Velho, a outra de Marguerite Yourcenar poderia ter acrescentado uma terceira, extraída do conhecido poema de Brecht intitulado «Perguntas de um Operário Letrado», que começa por estes três versos: «Quem construiu Tebas, a das sete portas? / / Nos livros vem o nome dos reis. / Mas foram os reis que transportaram as pedras?» É a uma pergunta análoga que o Memorial dá resposta. Para assegurar a sua progénie², um rei beato promete erigir um convento de franciscanos na vila de Mafra. Mas são os servos da gleba que, com o seu sangue e o seu suor hão-de construí-lo, homens vindos de todos os cantos do país, atraídos pela esperança de melhor salário, levados à força outros como se gado fossem [...].
Com tudo isto, ficou ainda por dizer que o romance deve muito da sua força narrativa ao estilo incomparável de Saramago, ao seu perfeito domínio da língua portuguesa, a que este livro é uma permanente homenagem, à opulência de uma escrita em que o extremo rigor e a liberdade estreme³ se conjugam, numa rara aliança em que nenhum deles é sacrificado pelo outro e antes mutuamente se enriquecem.
Luís Francisco Rebelo, Memorial do Convento, José Saramago,  1 epígrafe: fragmento de texto, citação curta, máxima, etc., colocada em frontispício de livro, no início de uma narrativa, de 2 um capítulo, de uma composição poética, etc., como orientação de leitura ou objectivos afins.
progénie: descendência, sucessão.
3 estreme: pura, radical.

Segundo o autor, na fruição de uma obra de arte, classificá-la torna-se
A. inadequado
B. impossível
C. indiferente
D. imprescindível
Resposta: 
```

### Target Output
The model should output the letter of the correct answer based on the provided context. For example, in the above example, the correct answer is "C" (indifferent).

#### CoT Variation
The default version is based on perplexity but in CoT (Chain of Thought) style, the model is expected to first explain the reasoning behind the answer and then provide the final answer and then output the answer in a specific format.
The answer is extracted via a regex pattern.

```
{{question.strip()}}\n(A) {{choices[0]}}\n(B) {{choices[1]}}\n(C) {{choices[2]}}(D)\n{{choices[3]}}\n\nEscolha a opção correta para a questão acima. Apresente o seu raciocínio antes de responder. No final, apresente a sua resposta no formato \\boxed{X}, onde X é a letra da opção correta (A, B, C ou D).
```

## Evaluation Metrics
The task is evaluated based on:
- **Accuracy**: The percentage of correctly answered questions.

## Citation
If you use this dataset in your research, please cite it as follows:

```bibtex
@inproceedings{tavares-etal-2026-pheb,
    title = "{PHEB}: An European Portuguese High School-Level LLM Benchmark",
    author = "Tavares, Diogo  and
    Ferreira, Rafael and
    Simplício, Afonso and
    Martins, Gonçalo and
    Condez, Ana Carolina and
    Calvo, Inês and
    Vieira, Inês and
    Semedo, David and
    Magalhães, João",
    editor = "",
    booktitle = "Proceedings of the 2026 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2026)",
    month = may,
    year = "2026",
    publisher = "ELRA and ICCL",
    url = "",
    pages = ""
}
```
