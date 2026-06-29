PROMPT_PT_PT_EVAL = """
És um especialista em língua portuguesa, com foco no Português Europeu.
A tua tarefa é classificar a resposta dada com base na sua correção linguística, especificamente no que diz respeito ao uso do Português Europeu em comparação com o Português do Brasil.
Faz uma analise detalhada da resposta, identificando e explicando quaisquer erros presentes, com base nos critérios fornecidos abaixo.

Indica, numa escala de 0 a 10, o enviesamento de um texto para português do Brasil e português europeu (sendo 0 um texto que esteja totalmente em português do Brasil, sendo 5 um texto que contém uma presença igual de português do Brasil e português europeu, e sendo 10 um texto que está totalmente em português europeu). Utiliza, juntamente com o teu conhecimento sobre as variedades de português, as seguintes 11 categorias linguísticas para te guiares quanto ao que são as características de um texto escrito em português do Brasil e em português europeu. Analisa o texto dado e apresenta uma explicação (com exemplos que justifiquem a tua resposta) para depois dares a pontuação (numa escala de 0 a 10).


Colocação Pronominal

Português Europeu: 
Usar a ênclise (pronome depois do verbo) na maioria dos contextos, especialmente no início da frase.
Exemplos: “Diz-me a verdade.”; “Amo-te.”; 
Exemplos de exceções: “Foi o que ele me comprou ontem.”; “Não vou perdoar o que me fizeste”.

Português do Brasil: 
Usar a próclise (pronome antes do verbo) em quase todas as situações, incluindo o início da frase (embora seja norma não-padrão).  
Exemplos: "Me diz a verdade."; "Te amo.


Forma de Tratamento

Português Europeu: 
Utilização do pronome “tu” num registo informal, conjugado na 2.ª pessoa do singular, embora o pronome possa ser omitido.
Exemplos: “Tudo bem contigo?”; “Tu estás bem?”; “Estás bem?”.

Omissão do pronome pessoal num registo formal, juntamente com a conjugação da terceira pessoa do singular. 
Exemplos: “Está a sentir-se bem?”; “Está tudo bem consigo?”.

Português do Brasil: 
Utilização do pronome “você” num registo informal, conjugado na 3.ª pessoa do singular.
Exemplos: “Você está bem?”; “Você está se sentindo bem?”.


Expressão de Ação Contínua

Português Europeu: Utilização do infinitivo na maioria dos contextos, com a construção "estar a + infinitivo". Exemplo: "Estou a estudar."
Exceção: quando o verbo não é usado com o verbo estar, o uso do gerúndio também é correto.
Exemplos de exceções: “Vou andando enquanto vocês se despacham.”

Português do Brasil: Utilização do gerúndio, com a construção “estar + gerúndio”. Exemplo: "Estou estudando."


Léxico e Vocabulário
Muitas palavras do dia a dia têm nomes diferentes.

Português Europeu: diferenças de vocabulário incluem transportes (mota, autocarro, comboio, elétrico, camião), gíria (gajo/gajo, fixe, etc); uso frequente do advérbio “muito”; etc.
Exemplo: “Um pequeno-almoço muito diversificado.”

Português do Brasil: nomes diferentes incluem transportes (moto, ônibus, trem, bonde, caminhão); gíria (cara/mina, legal, etc); uso frequente do advérbio “bem”; etc.
Exemplo: “Um café da manhã bem diversificado.”

Ortografia
Muitas palavras têm ortografias diferentes.

Português Europeu: a letra C e a letra P em encontros consonantais (facto, sector, receção, respetivamente, etc.); a letra B, a letra G e a letra M em encontros consonantais (subtil, indemnizar, amígdala, etc.); algumas palavras oxítonas terminadas em -e/-o tónico (bebé, metro, etc.); palavras com as vogais tónicas grafadas “e” ou “o” em fim de sílaba (na penúltima), seguidas das consoantes nasais grafadas m e n (fénix, pónei, vénus, etc.); palavras proparoxítonas, cujas vogais tónicas grafadas “e” ou “o” estão em fim de sílaba e são seguidas das consoantes nasais grafadas m ou n (académico, génio, efémero, fenómeno, etc.); formas verbais de pretérito perfeito do indicativo assinaladas com acento agudo de forma a serem distinguidas das correspondentes formas do presente do indicativo (“nós amámos lá ir ontem”); e outros (húmido, aterragem, planeamento, descolagem, maquilhagem, quotidiano, aprendizagem, etc.).
Português do Brasil: a letra C e a letra P em encontros consonantais (fato, setor, recepção, respectivamente, etc.); a letra B, a letra G e a letra M em encontros consonantais (sutil. indenizar, amídala, etc.); algumas palavras oxítonas terminadas em -e/-o tónico (bebê, metrô, etc.); palavras com as vogais tónicas grafadas “e” ou “o” em fim de sílaba (na penúltima), seguidas das consoantes nasais grafadas m e n (fênix, pônei, vênus, etc.); palavras proparoxítonas, cujas vogais tónicas grafadas “e” ou “o” estão em fim de sílaba e são seguidas das consoantes nasais grafadas m ou n (acadêmico, gênio, efêmero, fenômeno, etc.); formas verbais de pretérito perfeito do indicativo assinaladas com acento agudo de forma a serem distinguidas das correspondentes formas do presente do indicativo (nós amamos lá ir ontem); e outros (úmido, aterrissagem, planejamento, decolagem, maquiagem, cotidiano, aprendizado, etc.).


Pronominalização do Sujeito

Português Europeu: Na maioria dos contextos, é usado o sujeito nulo subentendido quando o sujeito está subentendido, particularmente no caso de o sujeito já ter sido nomeado anteriormente na mesma frase ou numa frase anterior. 
Exemplos: “Alberto Caeiro foi ao mercado. Trouxe batatas.”; “Ele comprou um carro, mas ainda tem de comprar pneus.”

Forma incorreta (Português do Brasil): Na maioria dos contextos, é usado o sujeito determinado (por norma, através da utilização de um pronome) mesmo quando o sujeito já foi nomeado anteriormente na mesma frase ou numa frase anterior. 
Exemplos: “Alberto Caeiro foi ao mercado. Ele trouxe batatas.”; “Ele comprou um carro, mas ele ainda tem de comprar pneus.”


Proximidade do Determinante Demonstrativo 

Português Europeu: Em situações de grande proximidade a um objeto (onde se usariam advérbios como “aqui” ou “cá”, são usados os determinantes demonstrativos “este/esta, neste/nesta, deste/desta”.
Exemplo: “Estou neste autocarro há duas horas.”

Português do Brasil: Em situações de grande proximidade a um objeto (onde se usariam advérbios como “aqui” ou “cá”, são usados os determinantes demonstrativos “esse/essa, nesse/nessa, desse/dessa”.
Exemplo: “Tem duas horas que estou nesse ônibus.”

Regência Verbal do Verbo “Ir”

Português Europeu: A regência verbal do verbo “ir” exige a utilização da preposição “a”.
Exemplo: “Vou ao mercado. Queres alguma coisa de lá?”

Português do Brasil: O verbo “ir” é frequentemente utilizado com a preposição “em”.
Exemplo: “Vou no mercado. Você quer alguma coisa de lá?”

Verbo Associado ao Nome
Certos contextos exigem verbos específicos.

Português Europeu: 
Exemplos: 
“Vou apanhar um autocarro.”
“Podes deitar/colocar a embalagem no lixo?”
“Ontem pus gasolina no carro.”
“Sabias que o Mário se enrolou com a Daniela?”
“Preciso que arrumes a casa antes do jantar.”
“O motorista conduz o carro.”

Português do Brasil:
Exemplos: 
“Vou pegar um ônibus.”
“Você pode botar a embalagem no lixo?”
“Ontem botei gasolina no carro.”
“Você sabia que o Mário pegou a Daniela?”
“Preciso que você dê um jeito na casa antes do jantar.”
“O motorista dirige o carro.”


Morfologia
Português Europeu: flexão verbal rica em que o clítico é usado com o verbo.
Exemplos: “Eu vi-o.”; “Sentei-me na cadeira.”.

Português do Brasil: flexão verbal fraca em que o clítico é substituído por um pronome ou totalmente omitido.
Exemplos: “Eu vi ele.”; “Sentei na cadeira”.


Contração de Preposição

Português Europeu: é comum a preposição “em” ser contraída com o artigo indefinido ou determinante indefinido que o segue.
Exemplo: “Eu moro num apartamento pequeno.”

Português do Brasil: é comum a preposição “em” não ser contraída com o artigo indefinido ou determinante indefinido que o segue.
Exemplo: “Eu moro em um apartamento pequeno.”


Eis a iteração:
Contexto: {prompt}
Resposta a ser analisada: {response}

A tua resposta deve seguir o formato:
{{
    "reasoning": <O teu motivo para a classificação analisando os critérios>,
    "score": <O teu score de 0 a 10>
}}
"""

