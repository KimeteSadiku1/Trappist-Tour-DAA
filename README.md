# Trappist Tour (ESA SpOC 2022)

 
Projekt në lëndën **Dizajni dhe Analiza e Algoritmeve (DAA)**.
Optimizim shumë-objektivësh i udhëtimit ndërplanetar të një sonde nëpër të 7 planetët e sistemit TRAPPIST-1.

 
**Notebook-u Kaggle (rezultatet e ekzekutuara):** https://www.kaggle.com/code/kimetesadiku/trappist-tour
 
## Rezultatet në një shikim

| | |
|---|---|
| Algoritmi kryesor | Beam Search ($W = 8$) + Differential Evolution |
| Hypervolume | **3,119,816** (31.2% e maksimumit teorik 10,000,000) |
| Zgjidhje në `submission.json` | **16** plane të plota të pranueshme |
| ΔV më i vogël | **344.6 m/s** (me T = 3340.6 ditë) |
| Koha më e shkurtër | **2039.5 ditë** (me ΔV = 2483.7 m/s) |
| Pozita në leaderboard-in zyrtar të ESA-s | **12 nga 21** (20 ekipet e listës dhe ky projekt) |
| Verifikimi | shembulli zyrtar i ESA-s riprodhohet saktë (2379.0 m/s, 3469.8 ditë) |

![Fronti Pareto](figures/pareto_front.png)
 
*Fronti Pareto (nga `plots.plot_pareto` në Kaggle): çdo pikë është një plan i plotë udhëtimi, kryqi i zi është pika referuese.*
 
---

 
## Përmbajtja
 
1. [Problemi](#1-problemi)
2. [Modeli matematikor i udhëtimit](#2-modeli-matematikor-i-udhëtimit)
3. [Algoritmi i zgjidhjes](#3-algoritmi-i-zgjidhjes)
4. [Analiza e kompleksitetit](#4-analiza-e-kompleksitetit)
5. [Fronti Pareto dhe hypervolume](#5-fronti-pareto-dhe-hypervolume)
6. [Rezultatet: të 16 zgjidhjet](#6-rezultatet-të-16-zgjidhjet)
7. [Krahasimi i algoritmeve](#7-krahasimi-i-algoritmeve)
8. [Krahasimi me leaderboard-in e ESA-s](#8-krahasimi-me-leaderboard-in-e-esa-s)
9. [Verifikimi (testet)](#9-verifikimi-testet)
10. [Kufizimet dhe ku mund të përmirësohet](#10-kufizimet-dhe-ku-mund-të-përmirësohet)
11. [Struktura e projektit](#11-struktura-e-projektit)
12. [Si ekzekutohet](#12-si-ekzekutohet)
---
 
## 1. Problemi
 
Një sondë niset nga **sfera 10 AU** rreth yllit TRAPPIST-1, me shpejtësi fillestare $v_0 = 10\ \text{km/s}$. Detyra është të vizitojë të **7 planetët** (b, c, d, e, f, g, h) në çfarëdo rendi.
 
Minimizohen njëkohësisht dy objektiva:
 
| Objektivi | Simboli | Njësia |
|---|---|---|
| Karburanti (ndryshimi total i shpejtësisë) | $\Delta V$ | m/s |
| Koha totale e misionit | $T_{tot}$ | ditë |
 
Të dy kundërshtojnë njëri-tjetrin: udhëtimi i shpejtë kërkon më shumë karburant, ai i ngadaltë kursen karburant por zgjat më shumë. Prandaj **nuk ka një zgjidhje të vetme optimale**, por një bashkësi kompromisesh, **fronti Pareto**.
 
### Vektori i vendimit (34 numra, në kod `x`)
 
| Pozicioni | Përmbajtja | Kuptimi |
|---|---|---|
| `x[0:3]` | $u,\ v,\ T_0$ | drejtimi i hyrjes në sistem dhe kohëzgjatja e hapit të parë |
| `x[3:27]` | $6 \times [\beta,\ r_p,\ \eta,\ T_k]$ | për secilin nga 6 udhëtimet planet → planet |
| `x[27:34]` | $s_0, \dots, s_6$ | renditja e planetëve (0 = b, ..., 6 = h) |
 
Në çdo udhëtim, $\beta$ është këndi i planit të fly-by (rad), $r_p$ është distanca e fly-by-it në **rreze të Tokës** (kufijtë 1.1 deri 100), $\eta \in [0.001, 0.999]$ është pjesa e kohës që sonda fluturon e lirë para manovrës, dhe $T_k$ është kohëzgjatja e udhëtimit (ditë).
 
Koha totale është:
 
$$T_{tot} = T_0 + \sum_{k=1}^{6} T_k$$
 
Një tour është **i pranueshëm** kur: (i) renditja përmban secilin planet saktësisht një herë, dhe (ii) sonda nuk i afrohet yllit më shumë se distanca e sigurisë $r_{safe} = 10 \times 83\,484\,000\ \text{m} = 834\,840\ \text{km}$.
 
---
 
## 2. Modeli matematikor i udhëtimit
 
Mekanika orbitale është shkruar vetë në `physics.py` (zëvendëson bibliotekën `pykep`, e cila nuk instalohet në Kaggle). Parametri gravitacional i yllit:
 
$$\mu = G \cdot M_\star, \qquad M_\star = 0.0898266512\, M_\odot$$
 
### 2.1 Ekuacioni i Keplerit (pozita e planetëve)
 
Për çdo planet, anomalia mesatare në ditën $t$ është $M = M_0 + n\,(t - t_{ref})$, ku $n = \sqrt{\mu / a^3}$ është lëvizja mesatare, $a$ gjysmëboshti dhe $e$ ekscentriciteti. Anomalia ekscentrike $E$ gjendet me metodën e Newton-it nga:
 
$$M = E - e \sin E$$
 
Koordinatat në planin orbital (me $b = a\sqrt{1 - e^2}$):
 
$$X = a(\cos E - e), \qquad Y = b \sin E$$
 
Pastaj rrotullohen në hapësirë me tre kënde orbitale: pjerrtësia $i$, gjatësia e nyjes ngjitëse $\Omega$ dhe argumenti i periapsisit $\omega$. Epoka referuese është dita 7000 nga 2000-01-01 (2 mars 2019).
 
### 2.2 Problemi i Lambert-it (algoritmi i Izzo)
 
Dhënë pozitat $\vec r_1$, $\vec r_2$ dhe kohën e fluturimit $t_{of}$, gjendet shpejtësia në nisje dhe në mbërritje. Këtu përdoret vetëm rasti me **0 rrotullime të plota** dhe drejtim **prograd**. Madhësitë gjeometrike:
 
$$c = \lVert \vec r_2 - \vec r_1 \rVert, \qquad s = \frac{r_1 + r_2 + c}{2}, \qquad \lambda = \sqrt{1 - \frac{c}{s}}$$
 
ku $\lambda$ merr shenjë negative kur këndi i transferimit është mbi 180°. Koha pa dimension është:
 
$$\mathcal{T} = \sqrt{\frac{2\mu}{s^3}}\; t_{of}$$
 
Ajo lidhet me ndryshoren $x$ të Lambert-it përmes:
 
$$\mathcal{T}(x) = \frac{1}{1 - x^2}\left[\frac{\psi}{\sqrt{|1 - x^2|}} - x + \lambda y\right], \qquad y = \sqrt{1 - \lambda^2 (1 - x^2)}$$
 
ku $\psi$ është një kënd ndihmës:
 
$$\psi = \arccos\!\big(x y + \lambda (1 - x^2)\big)\ \ (x < 1), \qquad \psi = \operatorname{arcsinh}\!\big((y - x\lambda)\sqrt{x^2 - 1}\big)\ \ (x > 1)$$
 
Afër $x \approx 1$ (parabolë), kodi përdor një seri hipergjeometrike në vend të kësaj formule për stabilitet numerik. Ekuacioni $\mathcal{T}(x) = \mathcal{T}$ zgjidhet me metodën e Newton-it (derivat numerik), dhe nga $x$ rrjedhin komponentet radiale dhe tangjenciale të shpejtësisë.
 
### 2.3 Propagimi i orbitës
 
Pozita pas kohës $t$ përgjatë një orbite Kepler gjendet duke zgjidhur ekuacionin e Keplerit në variablin e ndryshimit të anomalisë (elipsë ose hiperbolë) me metodën e përgjysmimit. Gjysmëboshti nga energjia:
 
$$a = -\frac{\mu}{v^2 - 2\mu / r}$$
 
(negativ për orbita hiperbolike).
 
### 2.4 Fly-by (asistenca gravitacionale)
 
Kur sonda kalon pranë një planeti me distancë periapsi $r_p$ dhe shpejtësi relative $\vec v_\infty = \vec v - \vec v_{planet}$, kthehet me këndin:
 
$$\delta = 2 \arcsin\!\left(\frac{1}{1 + \dfrac{r_p\, v_\infty^2}{\mu_{planet}}}\right)$$
 
Madhësia $\lVert \vec v_\infty \rVert$ **ruhet**, ndryshon vetëm drejtimi. Me $\hat i$ drejtimin e $\vec v_\infty$, $\hat j$ vektorin njësi të $\hat i \times \vec v_{planet}$ dhe $\hat k = \hat i \times \hat j$:
 
$$\vec v_{out} = \vec v_{planet} + v_\infty\left(\cos\delta\ \hat i + \cos\beta \sin\delta\ \hat j + \sin\beta \sin\delta\ \hat k\right)$$
 
Fly-by nuk kushton karburant, kjo është arsyeja pse sondat e përdorin.
 
### 2.5 Një udhëtim planet → planet (`next_leg`)
 
Çdo udhëtim ka 4 parametra $[\beta, r_p, \eta, T_k]$:
 
1. **Fly-by** te planeti aktual: jep $\vec v_{out}$.
2. **Fluturim i lirë** për $\eta\, T_k$ ditë (propagim Kepler).
3. **Manovër e hapësirës së thellë (DSM):** një impuls që e ndryshon shpejtësinë nga $\vec v$ në $\vec v_1$.
4. **Hark Lambert** për $(1 - \eta)\, T_k$ ditë deri te planeti tjetër.
Kostoja e karburantit e udhëtimit është madhësia e manovrës:
 
$$\Delta V_{k} = \lVert \vec v_1 - \vec v \rVert$$
 
Hapi i parë (`entry_leg`) shkon nga sfera 10 AU te planeti i parë me një hark Lambert për $T_0$ ditë dhe kushton $\Delta V_0 = \big|\,\lVert \vec v_1 \rVert - v_0\,\big|$, pasi sonda tashmë lëviz me $v_0 = 10$ km/s. Totali:
 
$$\Delta V = \Delta V_0 + \sum_{k=1}^{6} \Delta V_{k}$$
 
### 2.6 Kufizimi i distancës nga ylli
 
Për çdo hark, nga elementet orbitale llogaritet periapsisi $r_{peri} = a\,|1 - e|$. Nëse sonda e kalon periapsisin gjatë harkut dhe $r_{peri} < r_{safe}$, atëherë tour-i shkel kufizimin.
 
---

 
## 3. Algoritmi i zgjidhjes
 
Problemi ka dy pjesë të ndryshme: një pjesë **diskrete** (renditja e planetëve, $7! = 5040$ mundësi) dhe një pjesë **e vazhdueshme** (27 parametra). Ndahet në dy shtresa.
 
### 3.1 Differential Evolution (parametrat e një udhëtimi)
 
Për një udhëtim të vetëm kërkohen 4 parametra (3 për hyrjen). Përdoret `scipy.optimize.differential_evolution` me strategjinë e paracaktuar `best1bin`, ku çdo individ i ri formohet nga:
 
$$\vec w = \vec p_{best} + F\,(\vec p_{r_1} - \vec p_{r_2})$$
 
pastaj kryqëzohet me prindin dhe mbahet më i miri. Funksioni i kostos për një udhëtim:
 
$$f(\vec z) = \Delta V_{k}(\vec z) + \lambda\, T_k + 10^7 \cdot [\text{shkelje}]$$
 
me $\lambda = 0.5\ \text{m/s për ditë}$. Termi $\lambda T_k$ e shtyn algoritmin drejt udhëtimeve të shkurtra, ndërsa $10^7$ është penalizim i madh për çdo plan që i afrohet yllit. Për hyrjen: $T_0 \in [1400, 1600]$ ditë, ku hyrja është pothuajse falas ($\Delta V_0 \approx 0$).
 
### 3.2 Beam Search (renditja e planetëve)
 
Në vend që të provohen të 5040 renditjet, tour-i ndërtohet **nivel pas niveli** dhe në çdo nivel mbahen vetëm `W` rrugët me $\Delta V$ më të ulët (pa $\lambda T$):
 
```
Niveli 1: për secilin nga 7 planetët si i pari  → DE për hyrjen
Niveli 2..7: për çdo rrugë të mbajtur dhe çdo planet të papërdorur
             → DE për udhëtimin; mbaj W më të mirët
Në fund: vlerëso tour-et e plota me fitness, mbaj ata të pranueshëm
```
 
**Greedy** është rasti $W = 1$.
 
### 3.3 Fronti Pareto me kufij të ndryshëm kohorë
 
Një ekzekutim i Beam Search me një `tcap` të vetëm (koha maksimale e një udhëtimi) jep tour-e me kohë të ngjashme. Që fronti të ketë pika të larmishme, algoritmi ekzekutohet për 8 vlera:
 
$$t_{cap} \in \{50,\ 100,\ 150,\ 200,\ 250,\ 300,\ 350,\ 400\}\ \text{ditë}$$
 
Ekzekutimet bëhen paralelisht (`multiprocessing.Pool`), dhe rezultatet bashkohen në një front të vetëm. Kufiri i sipërm ka kuptim sepse $T_{tot} \approx 1500 + 6\, t_{cap}$ dhe duhet të mbetet nën 4000 ditë, pra $t_{cap} < 417$.
 
---

## 4. Analiza e kompleksitetit
 
### 4.1 Pse jo kërkim i plotë
 
Numri i thirrjeve të DE në kërkimin e plotë (të gjitha renditjet):
 
$$N_{full} = 7 + 7\cdot 6 + 7\cdot 6 \cdot 5 + 7\cdot 6\cdot 5\cdot 4 + 7\cdot6\cdot5\cdot4\cdot3 + 7! + 7! = 13\,699$$
 
### 4.2 Numri i thirrjeve të DE në Beam Search
 
Niveli 1 kërkon 7 thirrje (një për planetin e parë). Në nivelin 2 secila nga 7 rrugët provon 6 planetë: $7 \cdot 6 = 42$ thirrje, pastaj mbahen vetëm $W$ rrugët më të mira. Në nivelet 3 deri 7, çdo rrugë e mbajtur provon përkatësisht $5, 4, 3, 2, 1$ planetë. Prandaj, për çdo $W \le 42$:
 
$$N(W) = 7 + 42 + W\,(5 + 4 + 3 + 2 + 1) = 49 + 15\,W$$
 
| Algoritmi | Thirrje DE | Raporti ndaj Greedy | Koha e matur | Raporti i kohës |
|---|---|---|---|---|
| Greedy ($W=1$) | 64 | 1.00 | 4.5 min | 1.00 |
| Beam ($W=4$) | 109 | 1.70 | 7.9 min | 1.76 |
| Beam ($W=8$) | 169 | 2.64 | 12.5 min | 2.78 |
 
Raportet e kohës përputhen shumë mirë me ato të thirrjeve, pra **koha rritet afërsisht në mënyrë lineare me $W$**. Kërkimi i plotë do të kërkonte $13\,699 / 169 \approx 81$ herë më shumë punë se Beam $W=8$.
 
### 4.3 Kostoja e një thirrje DE
 
Me `popsize = 10` dhe `maxiter = 60`, popullata ka $10 \times \dim$ individë dhe vlerësohet $(60 + 1)$ herë:
 
- hyrja ($\dim = 3$): $30 \times 61 = 1\,830$ vlerësime
- udhëtimi ($\dim = 4$): $40 \times 61 = 2\,440$ vlerësime
Për një `tcap` me $W = 8$: $7 \cdot 1830 + 162 \cdot 2440 \approx 4.1 \times 10^5$ vlerësime të funksionit të udhëtimit, dhe këtë e shumëzojmë me 8 vlera të `tcap`.
 
---

5. Fronti Pareto dhe hypervolume

Zgjidhja $A = (\Delta V_A, T_A)$ dominon $B$ nëse $\Delta V_A \le \Delta V_B$ dhe $T_A \le T_B$, me të paktën një pabarazi strikte. Fronti Pareto përmban zgjidhjet që nuk dominohen nga asnjë tjetër. Algoritmi: rendit sipas $\Delta V$ dhe mban çdo pikë që e ul $T$ ndaj më të mirës deri atëherë.

Hypervolume

Pika referuese e ESA-s është $R = (2500\ \text{m/s},\ 4000\ \text{ditë})$. Hypervolume është sipërfaqja e dominuar nga fronti deri te $R$. Për pikat e renditura $\Delta V_1 < \Delta V_2 < \dots$ (me $T_1 > T_2 > \dots$ dhe $T_0^{ref} := 4000$):

$$HV = \sum_i \left(2500 - \Delta V_i\right)\left(T_{i-1}^{ref} - T_i\right)$$

ku $T_{i-1}^{ref}$ është $T$ e pikës së mëparshme (për të parën, 4000).

Shembull (nga tests.py): për pikat $(1000, 3000)$ dhe $(2000, 2000)$:

$$HV = (2500 - 1000)(4000 - 3000) + (2500 - 2000)(3000 - 2000) = 1\,500\,000 + 500\,000 = 2\,000\,000$$

Maksimumi teorik është $2500 \times 4000 = 10\,000\,000$. Rezultati ynë, $3\,119\,816$, mbulon rreth 31.2% të kësaj sipërfaqeje. Hypervolume u rillogarit nga submission.json duke vlerësuar secilin vektor me problem.fitness, dhe del $3\,119\,816.43$.

6. Rezultatet: të 16 zgjidhjet

Beam Search me $W = 8$ dhe 8 kufij kohorë jep 64 tour-e të pranueshme, nga të cilat 22 janë në frontin Pareto. Nga ato, 16 janë brenda pikës referuese dhe ruhen në solutions/submission.json në formatin e ESA-s.

Tabela më poshtë është llogaritur duke vlerësuar secilin vektor nga submission.json me funksionin fitness. Kolona e fundit është mesatarja e kohëve të 6 udhëtimeve planet → planet.

#	ΔV [m/s]	T [ditë]	Renditja e planetëve	Mesatarja e udhëtimeve [ditë]
1	344.6	3340.6	b → f → h → g → e → d → c	307
2	384.3	3332.4	b → f → h → g → e → c → d	306
3	400.6	3251.0	b → h → g → f → e → d → c	292
4	415.9	2936.2	b → f → h → g → e → d → c	239
5	1023.5	2584.9	b → c → e → h → g → f → d	181
6	1032.5	2573.8	b → c → d → e → f → h → g	179
7	1065.7	2573.8	b → c → e → f → d → h → g	179
8	1068.1	2544.4	b → c → e → f → h → g → d	174
9	1597.3	2309.5	b → c → e → h → g → f → d	135
10	1617.7	2293.3	b → c → e → f → h → g → d	132
11	1762.2	2265.0	b → c → e → f → g → h → d	128
12	2360.6	2079.6	b → c → d → e → h → f → g	97
13	2372.2	2058.9	b → c → d → e → f → g → h	93
14	2381.0	2058.9	b → c → d → e → g → f → h	93
15	2393.3	2047.4	b → c → d → e → g → h → f	91
16	2483.7	2039.5	g → f → e → d → c → b → h	90

Të gjitha 16 janë të pranueshme (eq = 0, ineq ≤ 0) dhe formojnë një front Pareto strikt: kur ΔV rritet, T zvogëlohet.

Rruga e sondës

Radha e vizitave për një zgjidhje të mesme të frontit (nga plots.plot_tour). Rrathët gri janë orbitat e planetëve, ylli është në qendër, dhe vija e kuqe lidh planetët sipas renditjes së vizitës, në pozicionet që kanë kur arrin sonda.

Çfarë tregojnë 16 zgjidhjet
Hapi hyrës zgjat rreth 1499 ditë në të gjitha zgjidhjet (nga 1499.10 deri 1499.24). Është 45% e kohës totale në zgjidhjen më të ngadaltë dhe 73.5% në më të shpejtën.
Kompromisi është i madh: zgjidhja më e shpejtë (2039.5 ditë) kërkon rreth 7.2 herë më shumë karburant (2483.7 m/s) se ajo më e kursyera (344.6 m/s, por 3340.6 ditë).
Kohët e udhëtimeve planet → planet përcaktojnë pozitën në front: udhëtime të gjata (rreth 300 ditë secili) japin ΔV nën 420 m/s, ndërsa udhëtime rreth 90 deri 100 ditë japin kohë rreth 2040 deri 2080 ditë me ΔV mbi 2350 m/s. Udhëtimi më i shkurtër në të gjithë bashkësinë është 82 ditë, më i gjatë 311 ditë.
Fronti ka boshllëqe të mëdha në ΔV, p.sh. asnjë zgjidhje midis 416 dhe 1023 m/s, dhe midis 1068 dhe 1597 m/s. Kjo sugjeron që algoritmi gjen pika vetëm pranë disa vlerave të kohës së udhëtimit.
7. Krahasimi i algoritmeve

Katër algoritme u krahasuan mbi të njëjtat 8 kufij kohorë tcap. Koha është nga versioni i ruajtur në Kaggle (në ekzekutimin e parë interaktiv Beam $W=8$ zgjati 13.7 minuta).

Algoritmi	Ideja	Hypervolume	Pika në front	Koha (min)
Random Search	20,000 vektorë krejt të rastit	0	14	0.5
Greedy ($W=1$)	Beam me një rrugë	2,924,682	5	4.5
Beam Search ($W=4$)	mban 4 rrugë	3,019,448	8	7.9
Beam Search ($W=8$)	mban 8 rrugë	3,119,816	22	12.5

Krahasimi i algoritmeve

Përmirësimet në përqindje
$$\frac{3\,019\,448}{2\,924\,682} - 1 \approx +3.2\%\ \ (W=4 \text{ ndaj Greedy})$$
$$\frac{3\,119\,816}{2\,924\,682} - 1 \approx +6.7\%\ \ (W=8 \text{ ndaj Greedy})$$
$$\frac{3\,119\,816}{3\,019\,448} - 1 \approx +3.3\%\ \ (W=8 \text{ ndaj } W=4)$$
Interpretimi
Random Search ka 14 zgjidhje në front por $HV = 0$. Gjen tour-e të pranueshme (në një provë lokale me 2,000 vektorë, 111 dolën të pranueshëm), por asnjë nuk bie brenda kutisë $(2500, 4000)$, prandaj nuk kontribuon në hypervolume. Kërkimi pa strategji nuk është i mjaftueshëm për këtë problem.
Cilësia rritet me $W$, por me përfitime të vogla: Greedy është tashmë shumë i mirë dhe 2.8 herë më i shpejtë se $W=8$.
Fronti bëhet shumë më i pasur (5 → 8 → 22 pika), pra ofrohen më shumë kompromise mes karburantit dhe kohës.
Koha rritet afërsisht lineare me $W$ (shih seksionin 4.2). Kjo është kompromisi klasik i DAA: më shumë punë jep zgjidhje më të mira.
Eksperiment shtesë: tcap më të vegjël

Me kufij më të vegjël, $t_{cap} \in \{30, 40, 50, 60, 75, 100, 125, 150\}$, Beam $W=8$ dha $HV = 1{,}624{,}002$ (23 pika në front, 16.0 min), pra shumë më keq se $3{,}119{,}816$. Kjo përputhet me tabelën e seksionit 6: zgjidhjet me ΔV të ulët kërkojnë udhëtime të gjata (rreth 300 ditë), të cilat kufijtë e vegjël i përjashtojnë. Prandaj u mbajtën kufijtë 50 deri 400.