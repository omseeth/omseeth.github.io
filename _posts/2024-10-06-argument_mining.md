---
layout: post
title: Was ist Argument Mining?
date: 2024-10-06 10:05:00
description: Eine Einführung zum Forschungsfeld des Argument Minings
tags: argument mining NLP proposition conclusion 
categories: 
---

In Abschnitt **1** stelle ich eine Definition für Argumente vor und gehe auf das Argumentieren als kommunikatives Phänomen ein. In dem zweiten Abschnitt **2** erläutere ich Modellierungen von Argumenten. Abschließend skizziere ich in **3** eine Übersicht zu Argument Mining als Aufgabe der natürlichen Sprachverarbeitung.

## 1 Argumentieren als sprachliches und kommunikatives Phänomen

Ein Argument kann nach der *Stanford Encyclopedia of Philosophy* wie folgt definiert werden: "as a complex symbolic structure where some parts, known as the premises, offer support to another part, the conclusion" (Dutilh Novaes, 2022). Ein Argument ist also ein Gebilde, das mindestens aus zwei Teilen besteht -- wobei wir die Definition noch um ein Gebilde aus *Propositionen* (d.h. wahrheitsfähigen Aussagen) erweitern sollten. In einem Argument stehen nach der Definition mindestens zwei Propositionen insoweit zueinander, sodass eine der Propositionen in einer stützenden Beziehung zu einer anderen steht. Die letztere wird als Konklusion bezeichnet. Geläufig sind aber auch Bezeichnungen wie 'Behauptung' oder 'Standpunkt'.

Die Stützkraft einer Prämisse kann unterschiedlich definiert werden (Dutilh Novaes, 2022). Prämissen können die Wahrheit einer Konklusion garantieren; sie können sie wahrscheinlicher oder akzeptabler machen. Auf der Grundlage dieser Einsicht können Argumente auch als deduktiv, induktiv, und abduktiv typisiert werden. Die erste umfangreiche Untersuchung zum deduktiven und induktiven Argumentieren wurde bereits in der Antike von Aristoteles (2007) vorgelegt. Deduktive Argumente führen zu einer Konklusion, die durch die Wahrheitswerte der Prämissen garantiert werden kann. Bei induktiven Argumenten werden Prämissen meist in Form von Beobachtungen zur Regularitäten vorgelegt, die in die Zukunft gerichtete Konklusionen wahrscheinlich machen. Der Philosoph Charles Sanders Peirce führt in seinen Vorlesungen in den 1860er Jahren schließlich das abduktive Argument (1982) ein, das sich von einem induktiven insofern unterscheidet, als das bereits einige Beobachtungen eine Konklusion möglicherweise erklären können und sie auf diese Weise akzeptabel erscheinen lassen.

Der Austausch von Argumenten kann als Argumentieren bezeichnet werden. Es ist eine dialogische Praxis, bei der meist die Behauptung einer Aussage zur Folge hat, dass diese weiter explizit mit Prämissen gestützt werden muss, um potentiell von einem Dialogpartner akzeptiert zu werden. Während einer Argumentation lässt sich beobachten, dass Prämissen auch genutzt werden, um Behauptungen zu untergraben. Allgemein wird beim Argumentieren vorausgesetzt, dass alle DialogpartnerInnen rational und ernsthaft an einem Austausch interessiert sind. Gleichzeitig ist die Sprache der Argumente nach Stede und Schneider (2019) oft subjektiv geprägt. (Dies mag verwundern, wenn zumindest der Anspruch eines ernsthaften Argumentierens eine inter-subjektive Übereinkunft ermöglichen sollte.)

Für Dutilh Novaes (2022) gibt es drei Ziele des Argumentierens. Zum einen können Meinungsverschiedenheiten durch Argumente aufgelöst werden, das heißt das Argumentieren dient dazu, einen Konsens herbeizuführen (1). Es kann jedoch zum anderen in einer rein adversarialen Auseinandersetzung verhaftet bleiben, zum Beispiel wenn politische KandidatInnen in öffentlichen Debatten ausschließlich eine Zuhörerschaft von ihrem Standpunkt überzeugen wollen (2). Nach Dutilh Novaes (2022) wird in den Wissenschaften das Argumentieren auch als eine epistemische Praxis eingesetzt, mit der Absicht, das Wissen aller Beteiligten zu erweitern (3). 

Der Verweis auf das tatsächliche Argumentieren deutet in eine Richtung, bei der sich klassische Konzepte der Argumentation von der kommunikativen Praxis unterscheiden. Argumente sind keinesfalls immer gut durchdacht und erfolgreich in ihrer Zielsetzung, ein Gegenüber zu überzeugen. Ein Argument, das weder wahrheitserhaltend noch einen Standpunkt wahrscheinlich oder akzeptabel macht, wird als 'Fehlschluss' bezeichnet. In ihrer *Introduction to Logic* stellen Copi und Cohen (2005, S. 126-7) mindestens 17 Typen solcher Fehlschlüsse vor. Zum Beispiel wird häufig versucht, das Gegenüber durch das Hervorrufen von Mitleid von einer Behauptung zu überzeugen. 

## 2 Modellierung von Argumenten

Zur Modellierung von Argumenten haben sich im 20. Jahrhundert drei Ansätze etabliert, die jeweils unterschiedliche Schwerpunkte setzen: das Toulmin-Modell, die *Neue Rhetorik*, und eine Modellstruktur nach Dung (1995).

Das Toulmin-Modell (Toulmin, 2003) teilt Argumentkomponenten in sechs mögliche Typen auf: *Claim*, *Data*, *Warrant*, *Backing, *Qualifier*, *Rebuttal*. Dabei besteht jedes Argument aus mindestens einer Behauptung (*Claim*), die durch Annahmen (*Data*) gestützt wird, und zwar auf der Grundlage einer Art Garantie (*Warrant*), die Annahmen und Behauptung miteinander verbindet. Die Garantie kann noch durch zusätzliche Aussagen gestützt werden (*Backing*). Das Toulmin-Modell lässt auch widerlegende Einschübe (*Rebuttal*) und Qualifizierungen der Komponenten (*Qualifier*) zu. Der Fokus des Toulmin-Modells liegt nicht auf einen Schlusstyp (sowohl induktive als auch deduktive Argumente können mit dem Schema modelliert werden). Es begrenzt sich allerdings auf die interne Struktur *eines* Argumentes.

Die *Neue Rhetorik* nach Perelman und Olbrechts-Tyteca (1991) legt ihren Schwerpunkt auf die Überzeugungskraft von Argumenten. Entscheidender Faktor des Argumentierens sind für die AutorInnen die Zuhörerschaft und wie diese für einen Standpunkt rhetorisch gewonnen werden kann. Deshalb sollten Argumente der Rhetorik nach insbesondere anhand ihrer perlokutionären Beschaffenheit verstanden und bewertet werden.

Dung (1995) hingegen modelliert nicht die interne Struktur eines Argumentes, sondern wie mehrere Argumente zueinander stehen. Im Fokus seines Ansatzes sind Angriffsbeziehungen zwischen Argumenten. Dung (1995) führt auch eine neue Abstraktionsebene ein: Jedes Argument kann als Knoten eines Graphen interpretiert werden. Die Kanten des Graphen entsprechen Angriffen, wobei die Richtung der Kante zwischen zwei Argumenten auf das angegriffene Argument zeigt.

Eine offene Frage bleibt an dieser Stelle, inwieweit Argumente als Bäume abstrahiert werden können. Ein Baum ist ein spezieller Graph, bei dem alle Knoten zusammenhängen und die Kanten keine Kreise bilden. Ansätze zur Modellierung von Argumenten (Palau und Moens, 2009; Peldszus und Stede, 2013; Musi et al., 2018; Hewett et al., 2019), die beispielsweise auch Schnittmengen mit der *Rhetorical Structure Theory* (Mann und Thompson, 1988) haben, interpretieren Argumente oft als gewurzelte Bäume, bei denen die Konklusion der Wurzel entspricht und die übrigen Komponenten auf diese mit gerichteten Kanten verweisen. In einer Korpusanalyse zu Argumenten geben (Niculae et al., 2017, S. 986) wiederum an, dass um die 20\% ihrer analysierten Argumentstrukturen nicht durch Bäume dargestellt werden können. Auch Pietron et al. (2024) stellen hierarchische Modellierungen von Argumenten mit Bäumen infrage, da Argumente in der kommunikativen Praxis vielfältigere Strukturen aufweisen würden.

## 3 Argument Mining in der natürlichen Sprachverarbeitung

Das *Argument Mining* hat sich in der natürichen Sprachverarbeitung zu Beginn des 21. Jahrhunderts, insbesondere ab den 2010er Jahren, als ein Forschungsbereich entwickelt, bei dem die automatische Identifikation und Extraktion von Argumenten und deren Strukturen verfolgt wird. Die Datengrundlage bildet meist geschriebene Sprache, in der Argumente präsentiert werden, wobei weniger literarische Texte genutzt werden, sondern vornehmlich Essays zu kontroversiellen Themen oder Kommentare aus dem Internet zu meist politischen Themen.

Eine allgemein anerkannte Definition des *Argument Minings* gibt es in der Wissenschaftsgemeinschaft nicht. Palau und Moens (2009), Peldszus und Stede (2013), Persing und Ng (2016), Eger et al. (2017), Stede und Schneider (2019), oder Lawrence und Reed (2019) präsentieren jeweils unterschiedliche Definitionen. Ein sehr detaillierter Ansatz wird von \citeStede und Schneider (2019) vorgelegt. Demnach lässt sich *Argument Mining* in folgende sieben Aufgaben aufteilen (gekürzte Übersetzung aus Stede und Schneider, 2019, S.6-7):

- Identifizierung von argumentativem Text
- Segmentiertung des Textes in argumentative Diskurseinheiten
- Identifizierung der zentralen Behauptung
- Identifizierung der Rolle oder Funktion der Einheiten
- Identifizierung von Relationen zwischen den Einheiten
- Aufbau einer strukturellen Repräsentation
- Identifizierung von Typen und der Qualität der Argumentation

Eine prägnantere Definition legen Palau und Moens (2009, S. 5) vor. Ihrer Definition nach lässt sich *Argument Mining* in drei Aufgaben teilen: (1) Identifizierung von Argumenten im Text, (2) Identifizierung der internen Struktur der Argumente, und (3) Identifizierung von Interaktionen zwischen Argumenten. Während Stede und Schneider (2019) den Schwerpunkt ihrer Definition eher auf die Mikrostruktur eines Argumentes legen (Aufgaben 2.-5.), gehört für Palau und Moens (2009) die Makrostruktur zwischen mehreren Argumenten zu gleichen Teilen zum Aufgabenbereich des *Argument Minings*. Palau und Moens (2009) lassen wiederum eine qualitative Bewertung von Argumenten außen vor.

Es gibt auch keine Einheitlichkeit bei der Frage, was die 'Bausteine' des 'Argument Minings' sind. Bei den möglichen Argumentkomponenten gibt es sowohl Ansätze mit unterschiedlicher Granularität als auch welche, die die Rollen der Argumentkomponenten umdeuten. Palau und Moens (2009) und Feng und Hirst (2011) nutzen zum Beispiel den *Araucaria*-Korpus (Reed, 2006), bei dem Argumentkomponenten als Prämissen und Konklusionen repräsentiert werden. Peldszus (2014), der den vorläufigen *Microtext*-Korpus entwickelt, weist Argumentkomponenten noch besondere Rollen zu, die als Opponent und Proponent bezeichnet werden, und sich aus der dialogischen Form des Argumentierens mit einem Für und Wider ableiten. Stab und Gurevych (2014a) nutzen den *Persuasive Essay*-Korpus, nach welchem Argumente aus Prämissen, Behauptungen, und zentralen Behauptungen (*Major Claim*) bestehen. Für Rinott et al. (2015) bestehen argumentative Passagen aus einem Thema, einer Behauptung sowie kontextabhängiger Evidenz. Niculae et al. (2017) nutzen den CDCP-Korpus, der keine argumentativen Komponenten von anderen Propositionen unterscheidet, sondern die letzteren in fünf Klassen (*Fact*, *Policy*, *Reference*, *Testimony*, *Value*) aufteilt und unter diesen Beziehungen herstellt. (Dass Wertaussagen (*Policy*, *Value*) auch als Propositionen, das heißt Aussagen mit Wahrheitsgehalt gewertet werden, wird vorausgesetzt.) Ein hybrider Ansatz wird schließlich von Schaefer et al. (2023) vorgestellt, wonach Behauptungen weiter als *Fact*, *Policy*, und *Value*, und Prämissen als *Testimony*, *Statistics*, *Hypthetical Instance*, *Real-example* und *Common-ground* typisiert werden können.

Ein ebenfalls heterogenes Bild gibt es mit Hinsicht möglicher Relationen zwischen Argumentkomponenten. Palau und Moens (2009) geben keine konkreten Relationen an, sondern stützen sich auf eine Vielfalt von möglichen Strukturen nach Walton et al. (2008) (z.B. *fulfillment* oder *causal* Relationen, oder verschiedene Angriffsrelationen wie *Rebuttal* oder *Undercutter*), die sie implizit mit Regeln einer kontextfreien Grammatik einzufangen versuchen. Peldszus (2014) folgt in Teilen dem Toulmin-Modell und definiert fünf mögliche Argumentrelationen: *Support*, *Example*, *Rebuttal*, *Undercutter*, *Linked*. Für Stab und Gurevych (2014a) gibt es nur *Support* und *Attack*. Rinott et al. (2015) beschränken wiederum mögliche Relationen zwischen argumentativen Propositionen auf rein unterstützende. Ähnlich verfahren auch Park und Cardie (2018), für die es bei der *Support*-Relation noch eine Unterteilung in *Reason* und *Evidence* gibt.

In dem weiteren Verlauf werde ich für die drei Bereiche des *Argument Minings* nach der kompakteren Definition von Palau und Moens (2009) beispielhaft einige Ansätze vorstellen.

Die Identifizierung von Argumenten in einem Text kann über die Segmentierung von Texteinheiten in argumentative oder nicht-argumentative Passagen realisiert werden. Für dieses Ziel nutzen Ajjour et al. (2017) einen Ansatz, bei dem sie für jeden möglichen Token eines Textes ein entsprechendes Label ermitteln. Für ihren Ansatz verwenden sie Texte aus drei Korpora unterschiedlicher Domänen (Essays, Nachrichten, online Kommentare). Die Modelle ihrer Experimente unterscheiden sich, inwieweit sie umliegende Tokens für die Klassifikation mit berücksichtigen: Eine lineare *Support Vector Machine* mit *Features* klassifiziert nur jeweils einen Token für sich, ein *Conditional Random Field* für Sequenzen berücksichtigt hingegen umliegende Tokens innerhalb eines Fensters. Als drittes lassen Ajjour et al. (2017) in ihrem bidirektionalen *Long Short-Term Memory*-Modell die Informationen aller vorangegangen und nachfolgenden Tokens mit in die Labelprädiktion eines jeden Tokens einfließen.

Die Analyse der internen Struktur eines Argumentes kann über die Klassifikation von Argumentkomponenten- und relationen erreicht werden. Auf der Grundlage des *Persuasive Essay*-Korpus nutzen Stab und Gurevych (2014b) ebenfalls ein Modell basierend auf einer *Support Vector Machine*. Stab und Gurevych (2014b) entwickeln dafür umfangreiche *Feature*-Sets, die auf der Grundlage struktureller Merkmale wie Interpunktion oder Satzlänge, lexikalischer Merkmale wie *n-grams*, von Adverbien, syntaktischer Merkmale, einschlägiger Diskursmarker, oder kontextueller Merkmale wie umliegenden Sätze oder Anzahl von Nebensätzen eine Prädiktion machen. Im Falle der Komponenten klassifizieren sie, ob es sich um Prämisse, Behauptung, oder zentrale Behauptung, und im Falle der Relationen, ob es sich um *Support*-Relationen handelt.

Schließlich kann das Verhältnis von Argumenten zueinander aus einer Makroperspektive wie bei Dung (1995) untersucht werden. Carstens und Toni (2015) nehmen dafür eine Minimaldefinition eines Argumentes an, wonach bereits einzelne Sätze argumentativen Gehalt haben und damit als Argumente behandelt werden können. Unter dieser Annahme klassifizieren sie ausschließlich Relationen (*Support* und *Attack*) zwischen mehreren Argumenten. Dafür nutzen Carstens und Toni (2015) ein *Random Forest*-Modell mit einem selbst entwickelten Korpus. Carstens und Toni (2015) vertreten die Annahme, dass sich aus der Bestimmung einer Relation automatisch ableiten lässt, welche Sätze auch Argumente sind.

## Zusätzliche Resourcen

Ich klassifiziere Propositionen und deren argumentative Relationen mit BERT im folgenden Projekt: [Argument Mining with BERT classification](https://github.com/omseeth/AM_BERT_classification)

## Bibliographie

Ajjour, Y., Chen, W.-F., Kiesel, J., Wachsmuth, H., und Stein, B. (2017). Unit Segmentation of Argumentative Texts. In Habernal, I., Gurevych, I., Ashley, K., Cardie, C., Green, N., Litman, D., Petasis, G., Reed, C., Slonim, N., und Walker, V., Herausgeber, Proceedings of the 4th Workshop on Argument Mining, Seiten 118–128, Copenhagen, Denmark. Association for Computational Linguistics.

Aristoteles (2007). BAND 3/I.1 Analytica priora. Buch I. Akademie Verlag, Berlin.

Copi, I. und Cohen, C. (2005). Introduction to Logic. Pearson/Prentice Hall.

Dung, P. M. (1995). On the acceptability of arguments and its fundamental role in nonmonotonic reasoning, logic programming and n-person games. Artificial Intelligence, 77(2):321–357

Dutilh Novaes, C. (2022). Argument and Argumentation. In Zalta, E. N. und Nodelman, U., Herausgeber, The Stanford Encyclopedia of Philosophy. Metaphysics Research Lab, Stanford University, Fall 2022. Auflage.

Eger, S., Daxenberger, J., und Gurevych, I. (2017). Neural End-to-End Learning for Computational Argumentation Mining. In Barzilay, R. und Kan, M.-Y., Herausgeber, Proceedings of the 55th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), Seiten 11–22, Vancouver, Canada. Association for Computational Linguistics.

Feng, V. W. und Hirst, G. (2011). Classifying arguments by scheme. In Lin, D., Matsumoto, Y., und Mihalcea, R., Herausgeber, Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies,
Seiten 987–996, Portland, Oregon, USA. Association for Computational Linguistics.

Hewett, F., Prakash Rane, R., Harlacher, N., und Stede, M. (2019). The Utility of Discourse Parsing Features for Predicting Argumentation Structure. In Stein, B. und Wachsmuth, H., Herausgeber, Proceedings of the 6th Workshop on Argument Mining, Seiten 98–103, Florence, Italy. Association for Computational Linguistics.

Lawrence, J. und Reed, C. (2019). Argument Mining: A Survey. Computational Linguistics, 45(4):765–818.

Mann, W. C. und Thompson, S. A. (1988). Rhetorical Structure Theory: Toward a functional theory of text organization. Text - Interdisciplinary Journal for the Study of Discourse, 8(3):243–281.

Musi, E., Alhindi, T., Stede, M., Kriese, L., Muresan, S., und Rocci, A. (2018). A Multi-layer Annotated Corpus of Argumentative Text: From Argument Schemes to Discourse Relations. In International Conference on Language Resources and Evaluation.

Niculae, V., Park, J., und Cardie, C. (2017). Argument Mining with Structured SVMs and RNNs. In Barzilay, R. und Kan, M.-Y., Herausgeber, Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Seiten 985–995, Vancouver, Canada. Association for Computational Linguistics.

Palau, R. M. und Moens, M.-F. (2009). Argumentation mining: the detection, classification and structure of arguments in text. In Proceedings of the 12th International Conference on Artificial Intelligence and Law, ICAIL ’09, Seite 98–107, New York, NY, USA. Association for Computing Machinery.

Park, J. und Cardie, C. (2018). A Corpus of eRulemaking User Comments for Measuring Evaluability of Arguments. In Calzolari, N., Choukri, K., Cieri, C., Declerck, T., Goggi, S., Hasida, K., Isahara, H., Maegaard, B., Mariani, J., Mazo, H., Moreno, A., Odijk, J., Piperidis, S., und Tokunaga, T., Herausgeber, Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan. European Language Resources Association (ELRA).

Peirce, C. S. (1982). Writings of Charles S. Peirce: A Chronological Edition, Volume 5: 1884-1886. Indiana University Press

Peldszus, A. (2014). Towards segment-based recognition of argumentation structure in short texts. In Green, N., Ashley, K., Litman, D., Reed, C., und Walker, V., Herausgeber, Proceedings of the First Workshop on Argumentation Mining, Seiten 88–97, Baltimore, Maryland. Association for Computational Linguistics.

Peldszus, A. und Stede, M. (2013). From Argument Diagrams to Argumentation Mining in Texts: A Survey. Int. J. Cogn. Inform. Nat. Intell., 7(1):1–31.

Pietron, M., Olszowski, R., und Gomu lka, J. (2024). Efficient argument classification with compact language models and ChatGPT-4 refinements.

Perelman, C. und Olbrechts-Tyteca, L. (1991). The New Rhetoric: A Treatise on Argumentation. University of Notre Dame Press.

Persing, I. und Ng, V. (2016). End-to-End Argumentation Mining in Student Essays. In Knight, K., Nenkova, A., und Rambow, O., Herausgeber, Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Seiten 1384–1394, San Diego, California. Association for Computational Linguistics.

Reed, C. (2006). Preliminary results from an argument corpus. In Berm´udez, E. M. und Miyares, L. R., Herausgeber, Linguistics in the Twenty-first Century, Seiten 185––196. Cambridge Scholars Press.

Rinott, R., Dankin, L., Alzate Perez, C., Khapra, M. M., Aharoni, E., und Slonim, N. (2015). Show Me Your Evidence - an Automatic Method for Context Dependent Evidence Detection. In M`arquez, L., Callison-Burch, C., und Su, J., Herausgeber, Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, Seiten 440–450, Lisbon, Portugal. Association for Computational Linguistics.

Schaefer, R., Knaebel, R., und Stede, M. (2023). Towards Fine-Grained Argumentation Strategy Analysis in Persuasive Essays. In Alshomary, M., Chen, C.-C., Muresan, S., Park, J., und Romberg, J., Herausgeber, Proceedings of the 10th Workshop on Argument Mining, Seiten 76–88, Singapore. Association for Computational Linguistics.

Stab, C. und Gurevych, I. (2014a). Annotating Argument Components and Relations in Persuasive Essays. In Tsujii, J. und Hajic, J., Herausgeber, Proceedings of COLING 2014, the 25th International Conference on Computational Linguistics: Technical Papers, Seiten 1501–1510, Dublin, Ireland. Dublin City University and Association for Computational Linguistics.

Stab, C. und Gurevych, I. (2014b). Identifying Argumentative Discourse Structures in Persuasive Essays. In Moschitti, A., Pang, B., und Daelemans, W., Herausgeber, Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), Seiten 46–56, Doha, Qatar. Association for Computational Linguistics.

Stede, M. und Schneider, J. (2019). Argumentation Mining. Morgan Claypool, San Rafael (CA).

Toulmin, S. E. (2003). The Uses of Argument. Cambridge University Press, 2. Auflage.

Walton, D., Reed, C., und Macagno, F. (2008). Argumentation Schemes. Cambridge University Press.