---
layout: post
title: How SoftBank’s Pepper set a positive example for gender design in robotics
date: 2025-03-26 10:00:00
description: This text explores the assignment of gender to robots
tags: robotics robots genders identities development engineering design
categories: 
---

**Abstract** Research in the philosophy of technology has shown that norms and politics resurface in our artifacts, technologies, machines, etc. There is no exception for robots. For instance, the reproduction of gender norms is especially visible in robots. The purpose of this text is to discuss how gendering takes place in robotics and what can be done to openly embrace gender politics in a development environment. Based on a brief account of Judith Butler’s gender theory, as well as an analysis of SoftBank’s Pepper, I argue that designers, engineers, and programmers should not seek to simply neutralize gendered features of their constructions. Instead, they should consciously and critically re-gender robots. They will otherwise run the danger of reproducing naturalized and exclusive stereotypes. Finally, I identify several challenges for critical gendering in robotics, such as the questionable attribution of gender to robots, the danger of reproducing binary notions, the importance of transparency, the limits of troubleshooting methods, and the peril of shallow marketing.

## Introduction

Langdon Winner argues that artifacts have politics. They are, as products of human activities, “ways of building order in our world” (Winner 2020, p. 28). In other words, the decisions we make to create an artifact materialize in the resulting product. Winner claims, for example, that the overpasses arching over the highway from New York City to Long Island are so low hanging for a reason. Apparently, the architect, who oversaw these structures, Robert Moses, did not want to have racial minorities or the low-income classes to visit the retreats of the rich on Long Island. That is why he made sure that it was impossible for vehicles such as buses to pass under the bridges, which meant that only those people who could afford a car at the time had access to Long Island. In her book, *Feminism Confronts Technology*, Judy Wajcman (1991) critically expands Winner’s thesis from a feminist perspective. She outlines that many technologies are built by a small group of men, who are usually employed by big companies. Their technologies are hence imbued with men’s world views as well as with the aim of producing a profitable product.

Having realized to what extent technologies are political in these senses, it is reasonable to assume that gender politics has an impact on the development of robots, too. As with overpasses, the ordering of robots must depend on the views and ideas of their designers, programmers, engineers, and so on. This is especially the case with social and care robots whose appearance and behavior are significant for human-robot interaction. The creators – be it consciously or unconsciously – will incorporate into their design whatever they believe to be paramount for successful socializing. We should therefore ask how robots are gendered.$$^1$$   There is the danger that uncritical design and programming, or even deliberately exclusive gender conceptions, reproduce sexist ideas and outdated stereotypes.

I will begin with Judith Butler, R. W. Connell, and Jack Halberstam, and discuss some elements of gender as a construct (1). I will then attempt a qualitative gender analysis of SoftBank’s Pepper (2). Lastly, I will argue that it would be counterproductive to simply neutralize and defuse gendering in robotics because gender distinctions seem to be inescapable (3). For this reason, I will propose that genders need to be openly embraced by those who design, engineer, program, and build robots. For this endeavor, in the last section, I will identify several gender specific challenges for robotics (4). Here I will focus on the concrete building process discussing, for example, whether hardware and software components of robots might reinforce sex and/or gender binaries.

## A rough sketch of gender

According to R. W. Connell “[g]ender is a way in which social practice is ordered” (Connell 2005, p. 71). For Connell this practice specifically relates to body structures as well as to the reproductive area. For example, a person wearing hot pants might be associated with what in most Western societies is conceived as the ‘feminine’. The ‘masculine’ as well as the ‘feminine’ are, however, contingent constructions. There is no reasonable justification, according to Connell, as to why the one gender is associated with this and the other gender with that. As simple as this might seem, it has been particularly difficult for human beings to escape such arbitrary stereotypes.

One explanation for gender differences is offered by Judith Butler (2007, 2011). With Butler, we can specify that genders as practices, or as she says performances, ought not to be understood as deliberate acts where one wakes up in the morning and decides what gender one has. We should rather understand gender as a way of responding to reiterated interpellations by authorities such as teachers, clergymen, cultural icons, parents, in brief, by those who have defined the dominant discourse of what is endorsed or forbidden. In other words, the construction of gender is the result of a normative process that can be difficult to escape. 

Butler underlines that the maintenance of gender norms also involves violence. For the constraints established by norms produce not only the “domain of intelligible bodies, but […] as well a domain of unthinkable, abject, unlivable bodies” (Butler 2011, p. x). At the beginning of the 21st century those bodies that are intelligible are usually those which adhere to the binary notion of man and woman, having a heterosexual orientation. All bodies that do not conform to these standards – here one could think of transsexual bodies – have often been met with violence and oppression. For abject bodies are, explains Butler, no simple opposition to the intelligible ones. They are perceived as specters of the latter because they remind any conforming body “of its own impossibility” (Butler 2011, p. x).

Another key insight in Butler’s work is that the ongoing need for gender performances also makes subversive iterations as well as normative changes possible, given that with each repetition there is the chance to alter the course of the staged act. It is, for example, possible that men can also wear hot pants.

Theories of gender, especially those by Judith Butler, have been criticized in at least two ways. First, since for Butler there is no pre-discursive body, no natural sex, as some might say, people have criticized that her theory would underestimate the materiality of bodies. Jack Halberstam points out how “performativity as a theoretical rubric” could, for example, imply “in a transphobic way, that trans* gender is not real, material, authentic” (Halberstam 2018, p. 120). In her book, Bodies That Matter, Butler (2011) agrees, emphasizing that bodies should not be excluded from gender analyses. She explains that they are not less real; they need to be fed and washed, they feel pleasure and pain, and each body has its own material peculiarity. However, Butler insists that even if bodies have an undeniable material reality, it does not mean that we can altogether escape the discursive structures defining bodies.

In her later writings, Butler picks up the second critique, namely the fear that gender awareness has become an object for marketing. In the 1999 preface to her Gender Trouble, she expresses her concern by concluding that “subversive performances always run the risk of becoming deadening cliches through their repetition and, most importantly, through their repetition within commodity culture where ‘subversion’ carries market value” (Butler 2007, p. xxiii). Halberstam points out in a similar fashion that increased flexibility with respect to genders “may function as a part of new regulatory regimes” (Halberstam 2018, p. 18). Butler thus concludes that “to name the criterion for subversiveness will always fail” (Butler 2007, p. xxiii). Each historical, social, and cultural gender context needs its own evaluation and analysis. 

With this sketch in mind, even with subversive gender politics, new forms of gender will not be safe from processes of exclusion. The way through which our identities are constituted is determined by a process of differentiating between bodies becoming intelligible against the backdrop of non-intelligible others. This theory is of dialectical nature. Therefore, some sort of rejection of the other, be it only in terms of a differentiation, seems to be a precondition for any gender. It is impossible that there will one day be an end to gender distinctions as some, including Stefan Hirschauer (2013), have stipulated. 

Hirschauer’s argument for “undoing gender” is premised on the same assumption as Butler’s hopes for subverting gender. As gender is socially reinforced and individually performed, Hirschauer believes in the possibility of changing social and personal practice to create spaces where gender becomes irrelevant. However, if such spaces exist, they would be inherently fragile and vulnerable to the same authoritative interpellations that dominate mainstream culture. A critical stance on gender should therefore try to perpetually overcome these dynamics by dissolving those genders that sharply contradict with the putatively non-conforming ones. 

## Qualitative gender analysis of Pepper

Having realized the historical and power-imbued dimensions of genders, we have a basis for tracing gender constructions that feed into what Butler has called the intelligibility of bodies, that is, the process of discursively reproducing a certain type of gender politics. With Winner and Wajcman, we know that politics – and this includes gender politics – also has an impact upon our technologies. I would now like to attempt an analysis of how gender politics influenced the design and shape of the robot Pepper that was introduced by SoftBank robotics in 2014. 

There are, however, a few caveats that need to be mentioned beforehand. One difficulty in analyzing possible gendering apparent in robots lies in the danger of reinforcing prescriptive norms by saying, for example, that this is ‘masculine’ behavior or that looks ‘feminine’, because a critical approach would want to put exactly these attributes into question. For this reason, I think that an analysis should try to proceed in as neutral a way as possible by sticking to descriptions rather than prescriptions. Of course, this approach will also reach its limits as well and run the danger of disguising itself as unbiased when there is no view from nowhere. 

Another point is that the following analysis leaves much to be desired because the accessible information is limited. Most of the development and designing sites for commercially marketed robots are not open to a critical public. The same holds true for some of the behind-the-scene features of robots such as speech recognition software, image processors, preprogrammed movements, possible reactions, etc., which would of course be very interesting for such an analysis since all of these aspects can be tilted in one way or another. 

Now let me introduce Pepper. Pepper was 1.20 meters tall robot that weighed 28 kilograms and looked, according to one article, “like a person (except for the wheels where its legs should be)” (Glaser 2016). The manual introduces Pepper as a humanoid that “has some physical human resemblance […] but doesn't pretend to be a human” (SoftBank Robotics 2017). Pepper had manga-like eyes, a print-on mouth, curvy hips and a comparatively small waist. Pepper’s breast was covered with a touch display that featured symbols like Pepper’s name, emoticons, or images. 

The robot’s default configuration could be adjusted by users. For example, it was possible to provide the robot with custom movements. Beyond its touchscreen, the default Pepper communicated through expressive gestures as well as speech. In the U.S. market, its voice varied in pitch – ranging from high tones in phrases like “*Thank you. What a sweet thing to say.*” to deeper tones in other contexts. The robot offered three voice styles, described in the manual as neutral, joyful, and didactic (SoftBank Robotics, 2017). Pepper’s responses were accompanied by short ringtones and illuminated signals around its eyes and ears, indicating successful speech recognition. SoftBank Robotics recommended keeping the robot animated to avoid an uncanny appearance.

The robot’s responses in conversations seemed to be mostly humorous, entertaining, sometimes flirty, but also informative. In response to the question “tell me about yourself”, Pepper answered, for example: “*My name is Pepper. I’m a humanoid robot and I’m 1.20 meters tall. I was born at SoftBank robotics. You can keep on asking questions if you want!*” While answering other questions, Pepper explained that it would be called Pepper because this name was easy to remember and translated easily into other languages, or that it was four years old and from Paris. In response to the question: “You come from Paris?” Pepper answered: “*I’m originally from Paris. Ahh Paris!*” while raising its head, looking to the sky with one of the arms to its hips. If you asked Pepper how much it would cost, Pepper was quick to respond: “*If you ask me, I’m priceless.*” But Pepper’s range of answers was also limited. In response to: “Do you have any feelings” Pepper said: “*I don’t understand. How about a taco?*” Giving silly answers appeared to be Pepper’s default response when the robot did not know about the questions being asked. Overall, Pepper’s lingual capabilities were most likely powered by a rule-based language model that used automatic speech recognition and text to speech engines$$^2$$ with a limited lexicon.

Pepper also offered an option to connect to a custom speech module based on Google’s Dialogflow platform. This platform enabled the creation of a conversational agent that guided Pepper’s communication using Google’s language models. The manual provided several recommendations on structuring Pepper’s conversations, including the importance of offering compliments or congratulations to users to make them feel appreciated. It also suggested that Pepper should take the lead in interactions, as most users were unfamiliar with robots and might feel shy.

Since Pepper was built for a specific form of employment, we need to assume that the robot was designed to fit its working contexts, which according to Kovach (2018) were businesses, shopping-malls, hospitals and doctor’s offices. So if we say that the robot was meant to take over service jobs, Pepper was employed in a sector where, according to World Bank statistics, the workforce was (and still is) predominantly female.$$^3$$  In other words, Pepper’s occupation had a potential impact on the employment of women. Was this taken into consideration when the robot was built? It is difficult to tell. Marketers at SoftBank Robotics were meticulously careful to avoid presenting a uniform image of the robot’s purposes – there was neither clear hint at where the robot should work, nor were there any concrete gender attributes. For instance, there were no pronouns used in any of the commercials. The same holds true for most of the media attention the robot received. In her article for Wired, Glaser (2016) used the pronoun “it” while referring to Pepper. 

For these reasons, I would not be sure how to label Pepper, if I had to attribute a gender to the robot. It would be easier to link certain of Pepper’s features to some established images and gender stereotypes. Pepper’s leg with wheels looked like a long dress covering the robot’s feet. And Pepper’s broad hips and slim waist as well as the robot’s round eyes made one think of rather ‘feminine’ attributes. Also, the fact that robot had a screen on its breast, diverting one’s attention from the face to this part of the body was questionable. The robot’s manner of employing many highly accentuated gestures, as well as Pepper’s dry enumeration of facts, reminded one of ‘masculine’ stereotypes. The same applied to some other physical features; for example, the robot had no hair. For me Pepper ranged between being a pet (think of the name) and (given Pepper’s size, limited answers and somewhat playful responses) a child. Thus, all in all, Pepper appeared as a potpourri of gendered features.

## The conscientious gendering of robots

After outlining the concept of gender and identifying gendered traits in Pepper, we can agree that developing non-sexist technologies requires critically examining problematic stereotypes and gender norms to create new avenues for emancipation. However, with an emphasis on the dynamics of exclusion in genders, I believe that a critical approach should not aim to simply neutralize or erase gender. Gender distinctions will inevitably resurface in some form. Empirical studies show that the perceived connection between users and robots equipped with social cues – such as gender stereotypes – shapes human-robot interaction (Powers et al., 2005). For instance, in Powers et al.’s study, participants were less talkative when interacting with a robot displaying ‘male’ attributes than with one exhibiting ‘female’ traits. This suggests that the influence of gendered cues is often prevalent.

The inherent dynamism of intelligible gender identities suggests that non-intelligible, even abject, bodies will always be necessary. This dialectical construction of identities ultimately precludes the possibility of fully neutralizing gender. A normative goal of gender neutrality – such as creating spaces or robots where gender plays no role – would therefore be too naive. Instead, technologies should be continuously re-gendered, incorporating ever-evolving designs that reflect a critical engagement with gender. Similarly, Tanja Kubes argues that after de-gendering robots, redesigning them can unlock “an enormous potential for exploiting the emancipatory possibilities of sexbots” (Kubes 2019, p. 70). This ongoing exploration of possibilities should serve as a guiding design principle.

In fact, Pepper might already be a good example of how gendering in robotics can be done. Pepper’s design incorporates elements traditionally associated with both masculine and feminine traits. While the large eyes and high-pitched responses might be read as feminine, the lack of hair and neutral voice settings avoid clear gender assignment. This ambiguity may have been intentional, allowing SoftBank to market Pepper as universally relatable. However, without explicit confirmation from SoftBank, it remains unclear whether this design was a strategic decision or an unintended consequence of broader robotic trends.

## Challenges for critical gendering in robotics

To facilitate conscientious gendering in robotics, I have identified some challenges that further need to be considered. 

(A) Can robots have something like a real gender, given the complexity of the concept? At the outset, I think that one can easily underestimate to what extent genders are tied to the human condition to which, for example, we would also attribute freedom and intention. We could otherwise not change the course of our gender performances, nor could we even think of changing them. It would therefore be exaggerated to speak of robots having genders since contemporary models do not have anything similar to what we call freedom and will. As with each technology in principle, it would thus be more appropriate to say that robots are gendered. But in light of the fact that many roboticists have set their heart on the idea that their creations will one day act autonomously, and that they would like to make them more human-like, we need to consider whether this would require that robots are also equipped with (or whether their development will inevitably lead to robots with) capacities facilitating the possibility of them having genders. For robots without genders would be not human-like, and robots with something like freedom and will might inevitably assume a gender.

(B) Do robots with their hardware and software reproduce the still established binary of sex/gender? For one thing, the analysis of Pepper has shown that design and engineering can help introducing new gender representations. From this perspective, they answer would be: not necessarily. For another thing, I think that one could still arrive at the conclusion that robots are good metaphors for how there is something material, such as the hardware, on the one side and something mind-like, such as the software, on the other side. This, in turn, might lead one to believe in what gender theory has tried to criticize, namely, the traditional binaries of sex/gender, or the view that our bodies are physically sexed, like machines, and whatever we consider as gender is socially programmed, like software. But this of course misses how interrelated and interdependent bodies and socially constructed identities are.

(C) How important is transparency in helping to raise awareness for gendering? Based on my brief analysis of Pepper, it is evident that robotics developers should be required to disclose key aspects of the development process. This includes identifying those involved in creating the robot, clarifying the goals pursued, and detailing the design choices and implemented features. Additionally, the robot’s speech capabilities should be explicitly linked to the underlying communication models, which, in turn, should be accompanied by data statements, as recommended by Emily Bender and Batya Friedman (2018).
While transparency in development (such as disclosing speech models) is crucial for ethical accountability, excessive transparency in interactions may hinder user experience. Striking a balance – where users understand how robots are designed without losing the perception of spontaneous social engagement – is key to responsible design.

(D) Should there be some sort of methodological troubleshooting that erases unwanted biases? There seems to be no reason why there should be none. But even if we have some rough methodological frame, which could make sure, for example, that the development team is not too homogeneous, there is always the peril of falling back to complaisant repetitions, ending, as Butler said, as deadening clichés. Developers should always keep in mind that there is no one-for-all criterion for subversiveness. Gender consciousness is an ongoing political process, and this would also mean an ongoing reinvention of troubleshooting methods and critical assessments to deal with genders in technology.

(E) Is creative gendering in robotics only good marketing and, in that, is it ignoring the broader political and economic issues that are bound up with genders? As criticized by Butler or Halberstam, gender is not only the construction of a broader social and cultural process, but agendas around gender can also be usurped by for-profit purposes where diversity is utilized for soliciting new customers, or for concealing other exploitative structures. Critical gendering in robotics might therefore run the risk of supporting just that. In this respect, this is one of the reasons why I am not sure how to evaluate Pepper since Pepper was, after all, a robot manufactured by SoftBank, which intended to sell as many Peppers as possible. The prize range was between 2,000$ and 30,000$. Robotics is big business. And doing politics within the context of big business is a complicated affair.

## Conclusion

This analysis of gender's social construction and its implications for robotics argues against seeking gender neutrality. Examining SoftBank’s Pepper as a positive case study, I highlighted the potential for robots to challenge gender stereotypes, advocating instead for critical re-gendering in design. This approach is essential to prevent the resurgence of exclusive norms. Furthermore, I discussed the challenges that this entails, from attributing gender to robots and navigating binary notions to ensuring transparency and avoiding superficial marketing. Future research should focus not only on designing gender-diverse robots but also on ensuring that these representations serve emancipatory rather than commercial interests. Additionally, interdisciplinary collaboration – between roboticists, ethicists, and gender theorists – will be essential in shaping an inclusive technological landscape.

## Footnotes

$$^1$$ Except for few feminists in technology studies such as Judy Wajcman, Lucy Suchman, Cynthia Cockburn or Susan Omrod, who have questioned technology related gender issues from a somewhat broader perspective, most of the publications discussing concrete gendering in robotics have only emerged since the 2010s. (Robertson 2010, Alesich & Rigby 2017, Kubes 2019)

$$^2$$ Some documentation of the speech module can be found here: http://doc.aldebaran.com/2-4/family/pepper_technical/languages_pep.html

$$^3$$ 60% of all women working worldwide are employed in services (World Bank, 2020a), the ratio for developed countries is even higher: in 2019 about 83% of all employed women in Japan and 90% in the United States were employed in services (World Bank 2020b).


## Bibliography

Simone Alesich and Michael Rigby. 2017. Gendered Robots. Implications for Our Humanoid 
Future. In *IEEE Technology and Society Magazine*, June 2017, Vol.36(2), pp.50-59.

Emily M. Bender and Batya Friedman. 2018. Data Statements for Natural Language 
Processing: Toward Mitigating System Bias and Enabling Better Science. *Transactions of the Association for Computational Linguistics*, 6:587–604.

Judith Butler. 2007. *Gender Trouble*. New York, Routledge.

Judith Butler. 2011. *Bodies That Matter*. New York, Routledge.

R. W. Connell. 2005. *Masculinities*. Cambridge, Polity Press.

April Glaser. 2016. Pepper, the Emotional Robot, Learns How to Feel Like an American. In 
*Wired*, viewed 14 April 2020, <https://www.wired.com/2016/06/pepper-emotional-robot-learns-feel-like-american/>.

Jack Halberstam. 2018. *Trans*. A Quick and Quirky Account of Gender Variability*. Oakland, 
University of California Press.

SoftBank Robotics. 2017. *How to Create a Great Experience with Pepper*.

Stefan Hirschauer. 2013. Die Praxis der Geschlechter(in)differenz und ihre Infrastruktur. In 
Julia Graf, Kirstin Ideler, Sabine Klinger (eds.) *Geschlecht zwischen Struktur und Subjekt: Theorie, Praxis, Perspektiven*. Opladen, Verlag Barbara Budrich, pp. 153-172.

IEEE. 2020. ‘HRP-4C’. IEEE, viewed 15 April 2020, < https://robots.ieee.org/robots/hrp4c/>.

Steve Kovach. 2018). We Interviewed Pepper – The Humanoid Robot. In *Tech Insider*, viewed 
14 April 2020 <https://www.youtube.com/watch?v=zJHyaD1psMc>.

Tanja Kubes. 2019. Bypassing the Uncanny Valley: Sex Robots and Robot Sex Beyond 
Mimicry. In Mark Coeckelbergh, Janina Loh (eds.) *Feminist Philosophy of Technology*. Berlin, J.B. Metzler, pp. 59-73.

Aaron Powers, Adam Kramer, Shirlene Lim, Jean Kuo, Sau-lai Lee and Sara Kiesler (2005). 
Eliciting Information from People with a Gendered Humanoid Robot*. In *IEEE International Workshop on Robots and Human Interactive Communication*, pp. 158-163.

Jenifer Robertson. 2010. Gendering humanoid robots: Robo-sexism in Japan. In *Body and 
Society*, vol. 16, no. 2, pp. 1-36.

Judy Wajcman. 1991. *Feminism Confronts Technology*. Cambridge, Polity Press.

Langdon Winner. 2020. Do Artifacts Have Politics? In *The Whale and the Reactor*. Chicago, 
University of Chicago Press, pp. 19-39.

World Bank. 2020a. Employment in services, female (% of female employment) (modeled 
ILO estimate). In *World Bank*, viewed 15 April 2020 <https://data.worldbank.org
/indicator/SL.SRV.EMPL.FE.ZS>.

World Bank. 2020b. Employment in services, female (% of female employment) (modeled 
ILO estimate) - Japan, United States. In *World Bank*, viewed 15 April 2020, <https://data.worldbank.org/indicator/SL.SRV.EMPL.FE.ZS?contextual=default&locations=JP-US>. 
