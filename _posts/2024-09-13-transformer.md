---
layout: post
title: Transformer-Modelle wie BERT erklärt (German version)
date: 2024-09-13 16:40:16
description: Vom neuronalen Netz bis zu BERT, eine kontextualisierte Erklärung der Transformer-Architektur
tags: machine learning, neural nets, feed-forward, attention-mechanism, transformer, BERT, transfer-learning, German
categories: 
---

Um die theoretische Grundlage zum Verständnis eines BERT-Modells zu schaffen, umreiße ich in diesem Blogbeitrag einige Konzepte der Transformerarchitektur. Dazu diskutiere ich in **1.1** *feedforward* neuronale Netze. In Abschnitt **1.2** beschreibe ich rekurrente neuronale Netze mit einer Enkodierer-Dekodierer-Architektur und dem ersten Aufmerksamkeitsmechanismus. In **1.3** führe ich alle Elemente zusammen, um ein Transformer-Modell zu beschreiben. Abschließend gehe ich in **1.4** auf die Besonderheiten BERTs ein und führe das Konzept des Transferlernens ein.

## **1.1** *Feedforward* neuronale Netze

Abstrakt betrachtet ist ein neuronales Netz zunächst ein Ansatz, um eine Funktion $$f(X, \theta)$$ zu approximieren, durch die mit den Parametern $$\theta$$ die Eingabewerte $$X$$ abgebildet werden können (Goodfellow et al. 2016). Als Klassifikator würde ein Netz zum Beispiel mit den richtigen Parametern $$f(x)=y$$ vorhersagen, wobei $$y$$ einem Klassenlabel entspricht (Goodfellow et al. 2016).

Neuronale Netze bestehen aus einer Mehrzahl von verknüpften Funktionen, die mit Hilfe eines gerichteten kreisfreien Graphen eine Eingabe bis zur Ausgabe verarbeiten. Die jeweiligen Funktionen können auch als Schichten (*layers*) $$h_{i}$$ mit $$i \in N$$ und $$N$$ als entsprechende Tiefe des Netzes bezeichnet werden. Die letzte Schicht wird als Ausgabeschicht $$a$$ bezeichnet. Anstatt dass man jede Funktion einer Schicht als eine Abbildung eines Vektor auf einen Vektor betrachtet, sollte man die Funktionen eher als weitere Kompositionen von Einheiten verstehen, die parallel Vektoren auf Skalare abbilden (Goodfellow et al. 2016). Für diese Abbildungen werden nicht-lineare Aktivierungsfunktionen genutzt. Ein neuronales Netz wird als *feedforward* bezeichnet, wenn von der Eingabe bis zur Ausgabe des Informationsflusses keine Form von Feedback berücksichtigt wird (Goodfellow et al. 2016). 

Die Approximation der Parameter $$\theta$$ folgt dem typischen Schema des maschinellen Lernens aus drei Schritten für jede Trainingsinstanz.


1. Für $$x \in X$$ sagt das Netz einen Wert $$\hat{y}$$ vorher.
2.  Dieser Wert wird mit einer weiteren Funktion, einer Kostenfunktion, 'bewertet'. Die Kosten geben eine Information darüber ab, inwieweit sich die Vorhersage des Netzes von einem Soll-Wert unterscheidet (typische Kostenfunktionen sind z.B. *Mean Squared Error* oder *Binary Cross Entropy*). Durch den Einbezug eines Soll-Wertes kann hier auch vom überwachten Lernen gesprochen werden.
3. Um zukünftige Kosten zu senken, werden mit einem Optimierungsalgorithmus alle Parameter des Netzes mit Hinblick auf die Kostenfunktion angepasst. Dieser Algorithmus versucht die Kostenfunktion zu minimieren, indem das globale Minimum der Funktion unter Rückgriff der Parameters des Modells für eine Eingabe angenähert wird. Der letzte Schritt involviert in den klassischen Verfahren das Berechnen der Gradienten der Parameter, mit denen das Modell für die nächste Lernrunde entsprechend der Steigung verändert werden kann (z.B. mit dem *Gradient-Descent*-Algorithmus). In einem mehrschichtigen neuronalen Netz müssen dafür partielle Ableitungen für alle Parameter der verketteten Funktionen gefunden werden. Die Berechnung kann mit dem *Backpropagation*-Verfahren durchgeführt werden, das rekursiv die Kettenregel innerhalb eines Berechnungsgraphen ausführt und so die Ableitungen aller Gradienten findet.

## **1.2** Rekurrente Netze mit einer Enkodierer-Dekodierer-Architektur

Im Unterschied zu den *feedforward* Netzen werden bei rekurrenten neuronalen Netzen Informationen innerhalb einer Schicht mit den Zuständen $$h_{i}^{t}$$ (auch *hidden states* bzw. verborgene Zustände genannt) ausgetauscht. Jeder Zustand zum Zeitpunkt $$t$$ erhält nicht nur Informationen aus der Eingabe, sondern auch aus den Zuständen $$t-1$$, also aus $$h_{i}^{t-1}$$ (vgl. **Fig. 1**). Die Rekurrenz kann dabei prinzipiell ebenfalls von der Ausgabe zu jeder Zwischenschicht oder nur bei jeder Ausgabe vorgenommen werden.

{% include figure.liquid loading="eager" path="assets/img/rnn.png" class="img-fluid mx-auto d-block" width="40%" %}**Fig 1:** Status eines neuronalen Netzes ohne (*feedforward*) und mit Feed-back (rekurrent) sowie jeweils einer Eingabeschicht $$x$$, einer verborgenen Schicht $$h1$$ und einer Ausgabe $$a$$

Der Vorteil rekurrenter neuronaler Netze wie dem *Long Short-Term Memory*-Modell (kurz LSTM Hochreiter und Schmidhuber, 1997) liegt darin, insbesondere sequenzielle Daten wie zum Beispiel Sprache sehr gut modellieren zu können. Wie man mit Ferdinand de Saussure inspiriert pointieren kann: Die Bedeutung eines Wortes leitet sich aus dem Zusammenspiel der Differenzen der umliegenden Wörter ab (de Saussure, 1931). So kann auch ein neuronales Netz wenig Sinn aus einer isolierten Betrachtung eines jeden Wortes ableiten. Werden hingegen die Bedeutungen der umliegenden Worteingaben mit in einer Schicht eines rekurrenten Netzes einbezogen, das heißt insgesamt eine Sequenz, können dadurch komplexere Zusammenhänge abgebildet werden.

Mit LSTM-Modellen entwickeln Sutskever et al. (2014) eine Sequenz-zu-Sequenz-Architektur zur maschinellen Übersetzung. Übersetzungen können nicht immer wortwörtlich vollzogen werden. Aus diesem Grund ist es sinnvoll, dass ein Modell einen ganzen Satz berücksichtigt, bevor es eine potentielle Übersetzung vorhersagt. Diese Idee realisieren Sutskever et al. (2014), indem sie eine Architektur aus zwei Teilen, einem Enkodierer und einem Dekodierer (siehe **Fig. 2**), entwickeln (vgl. auch Cho et al. 2014).

{% include figure.liquid loading="eager" path="assets/img/seq2seq.png" class="img-fluid mx-auto d-block" width="90%" %}**Fig 2:** Sequenz-zu-Sequenz-Architektur mit Enkodierer und Dekodierer

Der Enkodierer besteht aus einem LSTM-Modell, dem Vektorrepräsentationen (auch *Embeddings* genannt) für die Wörter einer Eingabesequenz aus der Ursprungssprache zugeführt werden. Es werden *Embeddings* aus dem einfachen Grund verwendet, da neuronale Netze nur mit Zahlen und nicht mit Buchstaben operieren können. Die verborgenen Status dieser Eingaben werden daraufhin durch das Modell zu einem finalen Zustand $$c$$ zusammengeführt.
\begin{equation}
    c = q(\{h^{1},...,h^{T}\})
\end{equation}
wobei $$q$$ dem LSTM-Modell entspricht und $$T$$ der Länge der Eingabesequenz. Dieser Zustand wird dem Dekodierer übergeben. 

Der Dekodierer besteht auch aus einem LSTM-Modell, welches auf der Grundlage des übergebenen Zustandes Wort für Wort eine Übersetzung in der Zielsprache vorhersagt. Dabei werden jedes übersetzte Wort und der Enkodierer-Endzustand der ursprünglichen Eingabe $$c$$ regressiv dem Dekodierer so lange zugeführt, bis das Modell die Übersetzung abschließt: 
\begin{equation}
    p(\textbf{y})= \prod_{t=1}^{T}p(y_{t}|\{y_{1},...,y_{t-1}\},c)
\end{equation}

Während des Trainings des Modells werden dem Enkodierer Sätze aus der Ursprungssprache und dem Dekodierer deren Übersetzungen entsprechend eines Hyperparameters (z.B. mit *Professor Forcing* (Goyal et al., 2016)) gezeigt, wodurch die Gewichte beider Kodierer zusammen gelernt werden können.

Um die Qualität der Übersetzungen, insbesondere für lange Sequenzen, zu verbessern, führen  Bahdanau et al. (2014) einen Aufmerksamkeitsmechanismus ein. Die Schwäche der Architektur nach Sutskever et al. (2014) liegt darin, dass die zu übersetzende Eingabe in eine einzige Repräsentation $$c$$ gezwängt wird, mit welcher der Dekodierer eine Übersetzung finden muss. Allerdings spielen für eine Übersetzung nicht alle Wörter eines Satzes eine gleich große Rolle und auch kann die Beziehung unter den Wörtern variieren. Ob der Artikel in 'the annoying man' für 'l'homme ennuyeux' mit 'le' oder 'l´' übersetzt wird, hängt im Französischen beispielsweise davon ab, ob auf den Artikel ein Vokal folgt, gegebenenfalls mit einem stummen 'h' davor (Bahdanau et al. 2014). Bahdanau et al. (2014) entwickeln deshalb einen Mechanismus, der diesen Nuancen besser gerecht wird (vgl. **Fig. 3**).

{% include figure.liquid loading="eager" path="assets/img/attention_seq.png" class="img-fluid mx-auto d-block" width="60%" %}**Fig 3:** Aufmerksamkeitsgewichte für eine Eingabe mit Bezug auf die Ausgabe an der Position *i=2*

Die um Aufmerksamkeit erweiterte Architektur übermittelt dem Dekodierer statt $$c$$ für jede Eingabe kontextabhängige Zustände $$c_{i}$$:
\begin{equation}
    c_{i} = \sum_{t=1}^{T}a_{it}h^{t}
\end{equation}
Das Gewicht $$a_{it}$$ für jeden Zustand $$h^{t}$$ (in (Bahdanau et al. 2014) auch 'Annotation' genannt), wird wie folgt ermittelt:
\begin{equation}
    a_{it} = \frac{\exp(e_{it})}{\sum_{k=1}^{T}\exp(e_{ik})}
\end{equation}
wobei $$a_{it}$$ eine Normalisierung (ähnlich der *Softmax*-Funktion) für das Anpassungsmodell $$e_{it}$$ ist. Dieses Modell ist wiederum ein *Feedforward*-Netz mit einer einzelnen Schicht, das bewertet, wie gut die Eingabe zum Zeitpunkt $$t$$ mit der Ausgabe an Position $$i$$ übereinstimmt. Damit erhält insgesamt jede Eingabe $$x^{1}...x^{T}$$ eine eigene Menge an Aufmerksamkeitsgewichten, die in $$c_{i}$$ resultieren, einem Kontextvektor, der dem Dekodierer hilft für jede Eingabe die passende Ausgabe (z.B. 'l'homme') zu bestimmen.

## **1.3** Transformer-Modelle mit Selbstaufmerksamkeit

Die Transformerarchitektur (Vaswani et al., 2017) führt einige der zuvor genannten Elemente zusammen. Die Architektur verleiht dabei dem Aufmerksamkeitsmechanismus eine wesentlich größere Rolle und verzichtet auf rekurrente Strukturen.

Der Enkodierer der Transformerarchitektur besteht aus Ebenen mit jeweils zwei Komponenten, durch die die eingehenden Informationen verarbeitet werden (vgl. **Fig. 4**). Eingaben werden als erstes einer Schicht mit einem Selbstaufmerksamkeitsmechanismus parallel zugeführt, der in Vaswani et al. (2017) vorgestellt wird. Nachdem dieser Mechanismus angewendet wurde, werden die Informationen normalisiert und daraufhin einer weiteren Schicht mit einem *feedforward* neuronalen Netz übergeben. Die Verarbeitung der Eingaben findet auf dieser Schicht wiederum einzeln statt. Für den Zwischenschritt der Normalisierung werden Mittelwerte und Standardabweichungen nach dem Prinzip der *Layer Normalization* berechnet (Ba et al. 2016).

{% include figure.liquid loading="eager" path="assets/img/transformer_enkodierer.png" class="img-fluid mx-auto d-block" width="60%" %}**Fig 4:** Enkodierer eines Transformer-Modells

Wie bei den vorigen Sequenz-zu-Sequenz-Architekturen wird die Eingabe auch zunächst in *Embeddings* übersetzt. Die *Embeddings* werden aber zusätzlich mit einer Positionsenkodierung versehen, die über eine Frequenzdarstellung realisiert wird. Dies begründet sich wie folgt. Im Gegensatz zu den rekurrenten Ansätzen verarbeitet die Aufmerksamkeitsschicht eines Transformerkodierers eine Eingabesequenz auf einmal und zum Beispiel im Falle einer Übersetzung nicht Wort für Wort. Ohne eine zusätzliche Information zur Position einer jeden Eingabe innerhalb einer Sequenz würden den Kodierern die wichtige Information fehlen, wie die einzelnen Wörter aufeinander folgen.

Für diese Verarbeitung hält die Transformerarchitektur eine *Embedding*-Matrix für alle Vokabeln der Daten bereit, die in das Training eines Tranformer-Modells einfließen. Die Größe der Matrix entspricht der Anzahl der Vokabeln (sonst als Tokens bezeichnet, zu denen auch bswp. Satzzeichen zählen) Kreuz einer gewählten Dimension (also Vokabelanzahl x Dimension), mit der jedem Eingabetoken genau eine Zeile innerhalb der Matrix zugeordnet werden kann. Die Werte für die Tokentypen werden während der Initialisierung eines Transformer-Modells zufällig ausgewählt. Es sind jene Zeilen, die auch als *Embeddings* bezeichnet werden. Man kann auch sagen, dass jeder Tokentyp eine Vektorrepräsentation besitzt. Wobei diese Vektoren in einem weiteren Schritt mit der positionalen Enkodierung addiert so der Eingabe eine einzigartige Darstellung verleihen. Entscheidend ist, dass die *Embeddings* der Transformerarchitektur sich im Laufe eines Trainings auch verändern, d.h. entsprechend der Daten angepasst, also 'gelernt' werden können. Zusätzlich führt die Positionsenkodierung der Tokens innerhalb jeder Eingabesequenz für eine kontextsensitive Repräsentation der verarbeiteten Tokens.

Der Dekodierer der Transformerarchitektur folgt strukturell dem Enkodierer mit einem Unterschied. Er enthält eine weitere Selbstaufmerksamkeitsschicht. In dieser Schicht können Teile der Dekodierereingabe verdeckt werden (dies wird auch als *Masking* beschrieben). Wie bereits erwähnt lässt ein Transformer-Modell grundsätzlich eine Betrachtung aller Eingaben gleichzeitig zu. Für die Prädiktion zum Beispiel einer Übersetzungen muss der Dekodierer allerdings ohne die Information der Lösungen arbeiten -- die diesem im Laufe des Trainings in Teilen unmaskiert zugänglich waren. Durch das Verdecken bleiben dem Dekodierer nur die enkodierten Eingaben aus der Ursprungssprache sowie autoregressiv die bereits vorhergesagten Wörter, bis der Übersetzungsprozess terminiert.

Im Gegensatz zu dem Aufmerksamkeitsmechanismus nach Bahdanau et al. (2014) entwickeln Vaswani et al. (2017) einen Selbstaufmerksamkeitsmechanismus, den sie auch als 'skalierte Skalarprodukt-Aufmerksamkeit' beschreiben (Vaswani et al., 2017, S.3). Die für Transformer genutzte Selbstaufmerksamkeit kann vereinfacht zunächst mit der Operation aus (3) verglichen werden (Raschka et al., 2022). Wir können für einen kontextsensitiven Vektor $$z_{i}$$ einer Eingabe an der Stelle $$i$$ die Aufmerksamkeit wie folgt berechnen (Raschka et al., 2022):
\begin{equation}
    z_{i} = \sum_{j=1}^{T}a_{ij}x^{j}
\end{equation}
wobei $$a_{ij}$$ nicht mit einem Status $$h^{t}$$, sondern mit den Eingaben $$x^{j}$$ multipliziert wird, mit $$j\in{\{1...T\}}$$ einer Eingabesequenz der Länge $$T$$ (vgl. die Summer über alle $$x^{j}$$ in (5)). Im Unterschied zu Bahdanau et al. (2014) ist $$a$$ dabei keine Normalisierung von einfachen *feedforward* Netzen $$e_{ij}$$, sondern eine *Softmax*-Normalisierung über die Skalarprodukte $$\Omega$$ der Eingabe $$x^{i}$$ bezogen auf alle anderen Eingaben $$x^{1}...x^{T}$$ (Raschka et al., 2022):
\begin{equation}
    a_{ij} = \frac{\exp(\omega_{ij})}{\sum_{j=1}^{T}\exp(\omega_{ij})}
\end{equation}
mit (Raschka et al., 2022):
\begin{equation}
    \omega_{ij} = x^{(i)T}x^{j}
\end{equation}
Was wir hier gleichzeitig im Unterschied zum Aufmerksamkeitsmechanismus von Bahdanau et al. (2014) sehen, bei welchem die Aufmerksamkeit insbesondere die Ausgabe des Dekodierers (dort an Ausgabeposition *i*) mit einbezieht, ist, dass das Aufmerksamkeitsgewicht in (6) mit (7) sich auf die anderen Eingaben einer Sequenz bezieht. Eben aus diesem Grund ist es sinnvoll, von einer *Selbstaufmerksamkeit* zu sprechen. 

Zu dieser Darstellung der Aufmerksamkeit fügen Vaswani et al. (2017) eine weitere Veränderung für jede Eingabe $$x^{i}$$ hinzu, und zwar wird das Gewicht $$a$$ nicht mit $$x^{j}$$ multipliziert, sondern mit einem Wert $$v^{j}$$: 
\begin{equation}
    z_{i} = \sum_{j=1}^{T}a_{ij}v^{j}
\end{equation}
Denn Vaswani et al. (2017) überführen jedes $$x$$ in ein Tripel aus ($$q^{i}$$, $$k^{i}$$, $$v^{i}$$) mittels den Projektionsmatrizen ($$U_{q}$$, $$U_{k}$$, $$U_{v}$$). Die Idee dahinter entstammt dem *Information Retrieval*, das mit Abfrage-, Schlüssel-, Werttripeln arbeitet. Die Skalarprodukte des Selbstaufmerksamkeitsmechanismus für jede Eingabe werden deshalb in Vaswani et al. (2017) auch nicht mit (7) berechnet, sondern mit den Abfrage- und Schlüsselwerten (Raschka et al., 2022):
\begin{equation}
    \omega_{ij} = q^{(i)T}k^{j}
\end{equation}
Kurz: Neben der Aufmerksamkeit einer Eingabe $$x^i$$ gegenüber allen anderen Eingaben innerhalb einer Sequenz $$X$$ wird die Selbstaufmerksamkeit noch durch verschiedene Darstellungen aller umliegenden $$x^j \in X$$ in Form ihrer Abfrage-, Schlüssel-, Wertrepräsentationen berechnet.

Die Selbstaufmerksamkeitsgewichte werden abschließend mit der Dimension der Eingabe noch skaliert ($$\frac{\omega_{ij}}{\sqrt{d}}$$) und können $$h$$-Mal parallel berechnet, wobei $$h$$ einer gewählten Anzahl an Köpfen (auch *Attention Heads* gennant) entspricht. Vaswani et al. (2017) wählen $$h=8$$ Köpfe, deren Werte konkatiniert abschließend den *feedforward* neuronalen Netzen in den Kodierern weitergereicht werden. Dieses Vorgehen kann auch als '*Multi-head Attention*' bezeichnet werden. Die zusätzliche Skalierung begründen Vaswani et al. (2017, S. 4) mit der Beobachtung, dass zu große Werte der Skalarprodukte (vgl. (9)) die für die Normalisierung genutzte *softmax* Funktion in einen Bereich führen, der beim Lernen in sehr kleine Gradienten resultiert.

## **1.4** Transferlernen via BERT

Derweil die ursprüngliche Transformerarchitektur zur maschinellen Übersetzung entwickelt wurde, haben sich Transformer-Modelle bei anderen Aufgaben ebenfalls bewährt. Am bekanntesten sind große Sprachmodelle wie *Generative Pre-trained Transformer*-Modelle  (Radford et al., 2018) von OpenAI, die unter der Bedingung einer Eingabe mit einem Transformer-Dekodierer den nächsten *Token* einer Sequenz vorhersagen. Das *Bidirectional Encoder Representations from Transformers*-Modell (kurz BERT, Devlin et al., 2019) ist wiederum ein Transformer-Enkodierer. Das heißt mit BERT können keine neuen Wörter oder Sätze in einer Zielsprache *autoregressiv* generiert werden. BERT stellt dafür Enkodierungen bereit, mit deren Hilfe sich zum Beispiel Klassifikationsaufgaben lösen lassen.

Für BERT trainieren Devlin et al. (2019)  zunächst den Enkodierer eines Transformer-Modells vor dem Hintergrund zweier Aufgaben. Die erste Aufgabe des BERT-Trainings besteht aus einer maskierten Sprachmodellierung (*Masked Language Modelling*). Das Modell bekommt dabei Sätze gezeigt, bei denen es 15% zufällig ausgewählte Wörter, die verdeckt werden, vorhersagen muss. Die zweite Aufgabe besteht aus einer binären Klassifikation zweier Sätze und zwar mit dem Ziel vorherzusagen, ob diese aufeinander folgen oder nicht. Wobei das Modell 50% korrekte Satzfolgen und 50% inkorrekte Folgen gezeigt bekommt. Da das Modell ausschließlich einen Enkodierer verwendet und keine Maskierung wie bei einem Transformer-Dekodierer vorgenommen wird, kann das Training auch als bi-direktional bezeichnet werden. Devlin et al. (2019) bezeichnen das Training ihres Enkodierers auf den beiden Aufgaben außerdem als 'Vortraining'.

In einem weiteren Schritt nutzen Devlin et al. (2019) ihr vortrainiertes BERT-Modell für weitere Experimente aus der natürlichen Sprachverarbeitung. Dafür feintunen sie BERT beispielsweise, um die plausibelste Wortfolge für Sätze aus dem Datensatz *Situations With Adversarial Generations* (Zellers et al., 2018) vorherzusagen. Da BERT ursprünglich vor dem Hintergrund bestimmter Aufgaben trainiert wurde, die Gewichte des Modells jedoch auch für neue Aufgaben verwendet und angepasst werden können, kann eine solche Verwendung BERTs auch als eine Form des Transferlernens bezeichnet werden.

## Bibliographie

Ba, J., Kiros, J.R., & Hinton, G.E. (2016). Layer Normalization. ArXiv, abs/1607.06450.

Bahdanau, D., Cho, K., und Bengio, Y. (2014). Neural Machine Translation by Jointly
Learning to Align and Translate. *CoRR*, abs/1409.0473.

Cho, K., van Merri ̈enboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., und Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. In Moschitti, A., Pang, B., und Daelemans, W., Herausgeber, *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, Seiten 1724–1734, Doha, Qatar. Association for Computational Linguistics.

de Saussure, F. (1931). *Cours de Linguistique Generale*. Payot, Paris.

Devlin, J., Chang, M.-W., Lee, K., und Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Burstein, J., Doran, C., und Solorio, T., Herausgeber, *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, Volume 1 (Long and Short Papers), Seiten 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Goodfellow, I., Bengio, Y., und Courville, A. (2016). *Deep Learning*. MIT Press.

Goyal, A., Lamb, A. M., Zhang, Y., Zhang, S., Courville, A. C., und Bengio, Y. (2016). Professor Forcing: A New Algorithm for Training Recurrent Networks. In Lee, D., Sugiyama, M., Luxburg, U., Guyon, I., und Garnett, R., Herausgeber, Advances in *Neural Information Processing Systems*, Band 29. Curran Associates, Inc.

Hochreiter, S. und Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Comput.*, 9(8):1735–1780.

Radford, A., Narasimhan, K., Salimans, T., und Sutskever, I. (2018). Improving Language Understanding by Generative Pre-Training. Technical report, OpenAI.

Raschka, S., Liu, Y., und Mirjalili, V. (2022). *Machine Learning with PyTorch and Scikit-Learn: Develop machine learning and deep learning models with Python*. Packt Publishing.

Sutskever, I., Vinyals, O., und Le, Q. V. (2014). Sequence to sequence learning with neural networks. In *Proceedings of the 27th International Conference on Neural Information Processing Systems - Volume 2*, NIPS’14, Seite 3104–3112, Cambridge, MA, USA. MIT Press.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. u., und Polosukhin, I. (2017). Attention is All you Need. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., und Garnett, R., Herausgeber, *Advances in Neural Information Processing Systems*, Band 30. Curran Associates, Inc.

Zellers, R., Bisk, Y., Schwartz, R., und Choi, Y. (2018). SWAG: A Large-Scale Ad- versarial Dataset for Grounded Commonsense Inference. In Riloff, E., Chiang, D., Hockenmaier, J., und Tsujii, J., Herausgeber, *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, Seiten 93–104, Brussels, Bel- gium. Association for Computational Linguistics.

**Version 0.1**