# S1B transfer diagnosis: first five fixed examples per set

## t2, episode 1287256782801180009, matrix

Support (encoded separately):
```text
Caldwell knows Nigel.
Julie knows Joshua.
Barry knows Shane.
Shane knows Caldwell.
Todd knows Anthony.
Joshua knows Barry.
Hannah knows Ron.
Anthony knows Julie.
Chase knows Todd.
Chan knows Jo.
Ron knows Chan.
Jo knows Felix.
```

Read question:
```text
Starting from Caldwell, keep following the relation 'knows' until there is no further fact for that relation. Which person is reached?
Answer:
```

Target: ' Nigel'; token IDs [65816]; answer-position index 28 / token ':'.

Top five next tokens: [{"token_id": 10946, "token": " Jo", "logit": 17.125, "probability": 0.2127024382352829}, {"token_id": 9305, "token": " Ann", "logit": 16.625, "probability": 0.12901054322719574}, {"token_id": 41302, "token": " Chan", "logit": 16.125, "probability": 0.07824885100126266}, {"token_id": 50106, "token": " Joan", "logit": 16.0, "probability": 0.06905437260866165}, {"token_id": 20060, "token": " Allen", "logit": 15.875, "probability": 0.06094026565551758}]

## t2, episode 8111687887098378937, matrix

Support (encoded separately):
```text
Conte knows Allen.
Connor knows Conte.
Mark knows Harm.
Matthias knows Paige.
Marine knows Christopher.
Harm knows Boris.
Boris knows Marine.
Kathy knows Hugo.
Hugo knows Mark.
Christopher knows Bruce.
Carolina knows Kathy.
Paige knows Connor.
```

Read question:
```text
Starting from Conte, keep following the relation 'knows' until there is no further fact for that relation. Which person is reached?
Answer:
```

Target: ' Allen'; token IDs [20060]; answer-position index 28 / token ':'.

Top five next tokens: [{"token_id": 40068, "token": " Norman", "logit": 17.5, "probability": 0.1743539422750473}, {"token_id": 11044, "token": " Peter", "logit": 17.25, "probability": 0.1357869952917099}, {"token_id": 20060, "token": " Allen", "logit": 16.5, "probability": 0.06414123624563217}, {"token_id": 20042, "token": " Alexander", "logit": 16.375, "probability": 0.05660444125533104}, {"token_id": 64995, "token": " Booker", "logit": 16.25, "probability": 0.049953240901231766}]

## t2, episode 7494498776518826428, matrix

Support (encoded separately):
```text
Barber knows Lynn.
Drake knows Bray.
Sandra knows Corey.
Cord knows Esther.
Chan knows Kristen.
Dale knows Susan.
Esther knows Barber.
Ella knows Dale.
Kristen knows Sandra.
Kata knows Cord.
Corey knows Kata.
Susan knows Drake.
```

Read question:
```text
Starting from Barber, keep following the relation 'knows' until there is no further fact for that relation. Which person is reached?
Answer:
```

Target: ' Lynn'; token IDs [44538]; answer-position index 28 / token ':'.

Top five next tokens: [{"token_id": 53391, "token": " Caroline", "logit": 20.5, "probability": 0.6347130537033081}, {"token_id": 88154, "token": " Brittany", "logit": 18.375, "probability": 0.0758056715130806}, {"token_id": 66861, "token": " Leah", "logit": 18.375, "probability": 0.0758056715130806}, {"token_id": 66363, "token": " Alison", "logit": 17.0, "probability": 0.019166674464941025}, {"token_id": 8876, "token": " Christian", "logit": 16.875, "probability": 0.016914531588554382}]

## t2, episode 6244493409239580883, matrix

Support (encoded separately):
```text
Julie knows Sydney.
Harm knows Julie.
Hardy knows Otto.
Bowman knows Lore.
Otto knows Olivier.
Christine knows Gala.
Claude knows Christine.
Rosie knows Patty.
Sydney knows Rosie.
Olivier knows Bowman.
Lore knows Chambers.
Gala knows Hardy.
```

Read question:
```text
Starting from Rosie, keep following the relation 'knows' until there is no further fact for that relation. Which person is reached?
Answer:
```

Target: ' Patty'; token IDs [88245]; answer-position index 28 / token ':'.

Top five next tokens: [{"token_id": 50870, "token": " Martha", "logit": 17.375, "probability": 0.27740567922592163}, {"token_id": 61153, "token": " Paula", "logit": 16.25, "probability": 0.09006045013666153}, {"token_id": 75840, "token": " Markus", "logit": 15.875, "probability": 0.061897579580545425}, {"token_id": 46585, "token": " Ellen", "logit": 15.875, "probability": 0.061897579580545425}, {"token_id": 28373, "token": " Charlotte", "logit": 15.8125, "probability": 0.05814739316701889}]

## t2, episode 3285038830464699338, matrix

Support (encoded separately):
```text
Mona knows Viktor.
Joel knows Leonard.
Ole knows Gene.
Viktor knows De.
Gandhi knows Ole.
Abdullah knows Scott.
Xavier knows Mona.
Damon knows Castillo.
Lisa knows Damon.
Scott knows Lisa.
Gene knows Joel.
De knows Abdullah.
```

Read question:
```text
Starting from Joel, keep following the relation 'knows' until there is no further fact for that relation. Which person is reached?
Answer:
```

Target: ' Leonard'; token IDs [40854]; answer-position index 28 / token ':'.

Top five next tokens: [{"token_id": 1581, "token": " De", "logit": 15.6875, "probability": 0.35250774025917053}, {"token_id": 15017, "token": " Dar", "logit": 14.3125, "probability": 0.08912792056798935}, {"token_id": 41984, "token": " Noah", "logit": 14.25, "probability": 0.08372792601585388}, {"token_id": 36819, "token": " Bir", "logit": 14.0625, "probability": 0.06941289454698563}, {"token_id": 41508, "token": " Isaac", "logit": 13.4375, "probability": 0.03715404495596886}]

## t2, episode 1287256782801180009, soft

Support (encoded separately):
```text
Caldwell knows Nigel.
Julie knows Joshua.
Barry knows Shane.
Shane knows Caldwell.
Todd knows Anthony.
Joshua knows Barry.
Hannah knows Ron.
Anthony knows Julie.
Chase knows Todd.
Chan knows Jo.
Ron knows Chan.
Jo knows Felix.
```

Read question:
```text
Starting from Caldwell, keep following the relation 'knows' until there is no further fact for that relation. Which person is reached?
Answer:
```

Target: ' Nigel'; token IDs [65816]; answer-position index 28 / token ':'.

Top five next tokens: [{"token_id": 18246, "token": " Josh", "logit": 16.25, "probability": 0.16788096725940704}, {"token_id": 3757, "token": " John", "logit": 15.125, "probability": 0.05450296774506569}, {"token_id": 14325, "token": " Ron", "logit": 15.0, "probability": 0.04809870198369026}, {"token_id": 33121, "token": " Neil", "logit": 14.9375, "probability": 0.04518454894423485}, {"token_id": 23269, "token": " Jonathan", "logit": 14.9375, "probability": 0.04518454894423485}]

## t2, episode 8111687887098378937, soft

Support (encoded separately):
```text
Conte knows Allen.
Connor knows Conte.
Mark knows Harm.
Matthias knows Paige.
Marine knows Christopher.
Harm knows Boris.
Boris knows Marine.
Kathy knows Hugo.
Hugo knows Mark.
Christopher knows Bruce.
Carolina knows Kathy.
Paige knows Connor.
```

Read question:
```text
Starting from Conte, keep following the relation 'knows' until there is no further fact for that relation. Which person is reached?
Answer:
```

Target: ' Allen'; token IDs [20060]; answer-position index 28 / token ':'.

Top five next tokens: [{"token_id": 51094, "token": " Hugo", "logit": 14.5, "probability": 0.05270662531256676}, {"token_id": 11044, "token": " Peter", "logit": 14.4375, "probability": 0.04951329156756401}, {"token_id": 61695, "token": " Maggie", "logit": 14.25, "probability": 0.04104796051979065}, {"token_id": 36353, "token": " Bryan", "logit": 14.25, "probability": 0.04104796051979065}, {"token_id": 32312, "token": " Victor", "logit": 14.0625, "probability": 0.03402995318174362}]

## t2, episode 7494498776518826428, soft

Support (encoded separately):
```text
Barber knows Lynn.
Drake knows Bray.
Sandra knows Corey.
Cord knows Esther.
Chan knows Kristen.
Dale knows Susan.
Esther knows Barber.
Ella knows Dale.
Kristen knows Sandra.
Kata knows Cord.
Corey knows Kata.
Susan knows Drake.
```

Read question:
```text
Starting from Barber, keep following the relation 'knows' until there is no further fact for that relation. Which person is reached?
Answer:
```

Target: ' Lynn'; token IDs [44538]; answer-position index 28 / token ':'.

Top five next tokens: [{"token_id": 53391, "token": " Caroline", "logit": 15.9375, "probability": 0.1321626752614975}, {"token_id": 66861, "token": " Leah", "logit": 15.5625, "probability": 0.09083399176597595}, {"token_id": 75826, "token": " Ariel", "logit": 15.3125, "probability": 0.07074158638715744}, {"token_id": 66363, "token": " Alison", "logit": 15.0, "probability": 0.051755648106336594}, {"token_id": 69697, "token": " Tara", "logit": 14.25, "probability": 0.024447636678814888}]

## t2, episode 6244493409239580883, soft

Support (encoded separately):
```text
Julie knows Sydney.
Harm knows Julie.
Hardy knows Otto.
Bowman knows Lore.
Otto knows Olivier.
Christine knows Gala.
Claude knows Christine.
Rosie knows Patty.
Sydney knows Rosie.
Olivier knows Bowman.
Lore knows Chambers.
Gala knows Hardy.
```

Read question:
```text
Starting from Rosie, keep following the relation 'knows' until there is no further fact for that relation. Which person is reached?
Answer:
```

Target: ' Patty'; token IDs [88245]; answer-position index 28 / token ':'.

Top five next tokens: [{"token_id": 28373, "token": " Charlotte", "logit": 16.25, "probability": 0.088618703186512}, {"token_id": 42360, "token": " Lauren", "logit": 16.125, "probability": 0.07820573449134827}, {"token_id": 66363, "token": " Alison", "logit": 15.9375, "probability": 0.0648348331451416}, {"token_id": 32026, "token": " Michelle", "logit": 15.75, "probability": 0.05374996364116669}, {"token_id": 26518, "token": " Chelsea", "logit": 15.375, "probability": 0.0369417741894722}]

## t2, episode 3285038830464699338, soft

Support (encoded separately):
```text
Mona knows Viktor.
Joel knows Leonard.
Ole knows Gene.
Viktor knows De.
Gandhi knows Ole.
Abdullah knows Scott.
Xavier knows Mona.
Damon knows Castillo.
Lisa knows Damon.
Scott knows Lisa.
Gene knows Joel.
De knows Abdullah.
```

Read question:
```text
Starting from Joel, keep following the relation 'knows' until there is no further fact for that relation. Which person is reached?
Answer:
```

Target: ' Leonard'; token IDs [40854]; answer-position index 28 / token ':'.

Top five next tokens: [{"token_id": 41984, "token": " Noah", "logit": 15.5, "probability": 0.14954215288162231}, {"token_id": 85016, "token": " Alec", "logit": 15.1875, "probability": 0.10940737277269363}, {"token_id": 38743, "token": " Gabriel", "logit": 14.375, "probability": 0.04854922741651535}, {"token_id": 24986, "token": " Christopher", "logit": 14.3125, "probability": 0.04560777544975281}, {"token_id": 44218, "token": " Seth", "logit": 14.25, "probability": 0.04284454509615898}]

## t1, episode 1340232750529134996, matrix

Support (encoded separately):
```text
April visits Teresa.
Allison visits Curt.
Ole visits Dante.
Pompeo helps Esther.
Allison trusts Gerard.
Urban visits Rene.
Tobias knows Rose.
Cole knows Burnett.
Braun knows Edmund.
Allison helps Andy.
Deep visits Chelsea.
Whitney helps Michel.
Braun visits Carter.
April trusts Paige.
Sole knows Logan.
Allison knows Doctor.
```

Read question:
```text
Which person is linked from Cole by the relation 'knows'?
Answer:
```

Target: ' Burnett'; token IDs [98894]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 71454, "token": " Dalton", "logit": 16.375, "probability": 0.30796337127685547}, {"token_id": 31003, "token": " Lucas", "logit": 15.125, "probability": 0.08823297917842865}, {"token_id": 24909, "token": " Russell", "logit": 15.0, "probability": 0.07786533236503601}, {"token_id": 43046, "token": " Logan", "logit": 14.875, "probability": 0.06871591508388519}, {"token_id": 33121, "token": " Neil", "logit": 14.1875, "probability": 0.03455253317952156}]

## t1, episode 3897341016785275333, matrix

Support (encoded separately):
```text
Daly trusts Teresa.
Bert visits Rena.
Hugh trusts Jerry.
Anthony knows Gray.
Roy visits Ludwig.
Gilbert trusts Matthew.
Colin helps Chung.
Good visits Hunter.
Bert trusts Cameron.
Anthony trusts Till.
Drew helps Stephen.
Georg helps Peter.
Anthony visits Glenn.
Anthony helps Felix.
Megan visits Romeo.
Hugh knows Michel.
```

Read question:
```text
Which person is linked from Megan by the relation 'visits'?
Answer:
```

Target: ' Romeo'; token IDs [70192]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 70192, "token": " Romeo", "logit": 18.5, "probability": 0.4506467580795288}, {"token_id": 75826, "token": " Ariel", "logit": 18.25, "probability": 0.3509640693664551}, {"token_id": 77723, "token": " Jaime", "logit": 16.375, "probability": 0.05382208153605461}, {"token_id": 82592, "token": " Elijah", "logit": 16.25, "probability": 0.04749782010912895}, {"token_id": 64415, "token": " Ronaldo", "logit": 15.0, "probability": 0.013608354143798351}]

## t1, episode 3318731652019559225, matrix

Support (encoded separately):
```text
Norman visits Gibson.
Edmund trusts Josh.
Norman knows Tommy.
Pavel helps Bruce.
Gould helps Harold.
Edmund visits Terry.
Carson knows Gates.
Carson trusts Joshua.
Randy visits Benton.
Carson helps Dar.
Garrison trusts Malik.
Edmund helps Manuel.
Farmer knows Kirk.
Clifford visits Wolfgang.
Edmund knows Morgan.
Billy knows Nora.
```

Read question:
```text
Which person is linked from Clifford by the relation 'visits'?
Answer:
```

Target: ' Wolfgang'; token IDs [86498]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 86498, "token": " Wolfgang", "logit": 19.5, "probability": 0.8794119358062744}, {"token_id": 79735, "token": " Werner", "logit": 16.75, "probability": 0.05621892213821411}, {"token_id": 70038, "token": " Urs", "logit": 14.75, "probability": 0.007608403917402029}, {"token_id": 57806, "token": " Roland", "logit": 14.375, "probability": 0.005229174625128508}, {"token_id": 76016, "token": " Viktor", "logit": 14.25, "probability": 0.004614729899913073}]

## t1, episode 8789847747578686652, matrix

Support (encoded separately):
```text
Eric visits Christine.
Georg knows Lucky.
Burke helps Johnny.
Bauer knows Wayne.
Matthias knows Giovanni.
Chloe trusts Andres.
Michelle knows Matthew.
Stacy helps Paige.
Steven trusts Day.
Matthias helps Berger.
Betty helps Cannon.
Oswald helps Fernando.
Sue visits Gerard.
Mona trusts Falk.
Georg visits Stewart.
Georg trusts Parker.
```

Read question:
```text
Which person is linked from Eric by the relation 'visits'?
Answer:
```

Target: ' Christine'; token IDs [45893]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 45893, "token": " Christine", "logit": 17.625, "probability": 0.23405569791793823}, {"token_id": 74571, "token": " Cynthia", "logit": 17.375, "probability": 0.18228276073932648}, {"token_id": 69331, "token": " Cindy", "logit": 17.0, "probability": 0.12528099119663239}, {"token_id": 81348, "token": " Cathy", "logit": 16.375, "probability": 0.06705808639526367}, {"token_id": 65721, "token": " Iris", "logit": 16.25, "probability": 0.05917854979634285}]

## t1, episode 5368454428460118048, matrix

Support (encoded separately):
```text
Suzanne visits Adolf.
Shirley trusts Kirk.
Johnny visits Mara.
Silva helps Chambers.
Glenn trusts Clay.
Shirley visits April.
Suzanne knows Wayne.
Amber knows Buch.
Mohamed trusts Edmund.
Adler visits Bennett.
Shirley knows Malik.
Amber visits Baxter.
Shirley helps Frost.
Gaines trusts Roberto.
Gates knows Foley.
Leonardo helps Frank.
```

Read question:
```text
Which person is linked from Gaines by the relation 'trusts'?
Answer:
```

Target: ' Roberto'; token IDs [54083]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 54083, "token": " Roberto", "logit": 16.75, "probability": 0.38636767864227295}, {"token_id": 36926, "token": " Paulo", "logit": 15.375, "probability": 0.09768904000520706}, {"token_id": 83369, "token": " Paolo", "logit": 14.75, "probability": 0.05228917673230171}, {"token_id": 66854, "token": " Philippe", "logit": 14.5625, "probability": 0.0433492474257946}, {"token_id": 54400, "token": " Rafael", "logit": 14.25, "probability": 0.03171498700976372}]

## t1, episode 1340232750529134996, soft

Support (encoded separately):
```text
April visits Teresa.
Allison visits Curt.
Ole visits Dante.
Pompeo helps Esther.
Allison trusts Gerard.
Urban visits Rene.
Tobias knows Rose.
Cole knows Burnett.
Braun knows Edmund.
Allison helps Andy.
Deep visits Chelsea.
Whitney helps Michel.
Braun visits Carter.
April trusts Paige.
Sole knows Logan.
Allison knows Doctor.
```

Read question:
```text
Which person is linked from Cole by the relation 'knows'?
Answer:
```

Target: ' Burnett'; token IDs [98894]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 43046, "token": " Logan", "logit": 16.625, "probability": 0.5313696265220642}, {"token_id": 71454, "token": " Dalton", "logit": 15.0, "probability": 0.10463286936283112}, {"token_id": 24909, "token": " Russell", "logit": 14.9375, "probability": 0.0982934907078743}, {"token_id": 31003, "token": " Lucas", "logit": 14.125, "probability": 0.04361746832728386}, {"token_id": 49623, "token": " Burke", "logit": 13.25, "probability": 0.01818246580660343}]

## t1, episode 3897341016785275333, soft

Support (encoded separately):
```text
Daly trusts Teresa.
Bert visits Rena.
Hugh trusts Jerry.
Anthony knows Gray.
Roy visits Ludwig.
Gilbert trusts Matthew.
Colin helps Chung.
Good visits Hunter.
Bert trusts Cameron.
Anthony trusts Till.
Drew helps Stephen.
Georg helps Peter.
Anthony visits Glenn.
Anthony helps Felix.
Megan visits Romeo.
Hugh knows Michel.
```

Read question:
```text
Which person is linked from Megan by the relation 'visits'?
Answer:
```

Target: ' Romeo'; token IDs [70192]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 70192, "token": " Romeo", "logit": 17.75, "probability": 0.4174540042877197}, {"token_id": 75826, "token": " Ariel", "logit": 17.625, "probability": 0.3684018850326538}, {"token_id": 82592, "token": " Elijah", "logit": 16.25, "probability": 0.09314657747745514}, {"token_id": 77723, "token": " Jaime", "logit": 15.125, "probability": 0.030240267515182495}, {"token_id": 64415, "token": " Ronaldo", "logit": 14.5, "probability": 0.01618644781410694}]

## t1, episode 3318731652019559225, soft

Support (encoded separately):
```text
Norman visits Gibson.
Edmund trusts Josh.
Norman knows Tommy.
Pavel helps Bruce.
Gould helps Harold.
Edmund visits Terry.
Carson knows Gates.
Carson trusts Joshua.
Randy visits Benton.
Carson helps Dar.
Garrison trusts Malik.
Edmund helps Manuel.
Farmer knows Kirk.
Clifford visits Wolfgang.
Edmund knows Morgan.
Billy knows Nora.
```

Read question:
```text
Which person is linked from Clifford by the relation 'visits'?
Answer:
```

Target: ' Wolfgang'; token IDs [86498]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 86498, "token": " Wolfgang", "logit": 19.875, "probability": 0.9652981162071228}, {"token_id": 79735, "token": " Werner", "logit": 15.25, "probability": 0.009463449008762836}, {"token_id": 57806, "token": " Roland", "logit": 14.3125, "probability": 0.003705939743667841}, {"token_id": 76016, "token": " Viktor", "logit": 13.6875, "probability": 0.001983646536245942}, {"token_id": 87936, "token": " Tomas", "logit": 13.375, "probability": 0.0014512669295072556}]

## t1, episode 8789847747578686652, soft

Support (encoded separately):
```text
Eric visits Christine.
Georg knows Lucky.
Burke helps Johnny.
Bauer knows Wayne.
Matthias knows Giovanni.
Chloe trusts Andres.
Michelle knows Matthew.
Stacy helps Paige.
Steven trusts Day.
Matthias helps Berger.
Betty helps Cannon.
Oswald helps Fernando.
Sue visits Gerard.
Mona trusts Falk.
Georg visits Stewart.
Georg trusts Parker.
```

Read question:
```text
Which person is linked from Eric by the relation 'visits'?
Answer:
```

Target: ' Christine'; token IDs [45893]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 45893, "token": " Christine", "logit": 18.625, "probability": 0.4588667154312134}, {"token_id": 74571, "token": " Cynthia", "logit": 17.125, "probability": 0.10238700360059738}, {"token_id": 65721, "token": " Iris", "logit": 17.0, "probability": 0.09035621583461761}, {"token_id": 69331, "token": " Cindy", "logit": 16.75, "probability": 0.07036948949098587}, {"token_id": 81348, "token": " Cathy", "logit": 16.375, "probability": 0.048364195972681046}]

## t1, episode 5368454428460118048, soft

Support (encoded separately):
```text
Suzanne visits Adolf.
Shirley trusts Kirk.
Johnny visits Mara.
Silva helps Chambers.
Glenn trusts Clay.
Shirley visits April.
Suzanne knows Wayne.
Amber knows Buch.
Mohamed trusts Edmund.
Adler visits Bennett.
Shirley knows Malik.
Amber visits Baxter.
Shirley helps Frost.
Gaines trusts Roberto.
Gates knows Foley.
Leonardo helps Frank.
```

Read question:
```text
Which person is linked from Gaines by the relation 'trusts'?
Answer:
```

Target: ' Roberto'; token IDs [54083]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 54083, "token": " Roberto", "logit": 17.5, "probability": 0.525161862373352}, {"token_id": 36926, "token": " Paulo", "logit": 16.125, "probability": 0.1327817291021347}, {"token_id": 54400, "token": " Rafael", "logit": 15.5, "probability": 0.07107293605804443}, {"token_id": 65194, "token": " Sergio", "logit": 15.125, "probability": 0.04884766414761543}, {"token_id": 66854, "token": " Philippe", "logit": 15.0, "probability": 0.04310791566967964}]

## t2_t1_wording, episode 1287256782801180009, matrix

Support (encoded separately):
```text
Caldwell knows Nigel.
Julie knows Joshua.
Barry knows Shane.
Shane knows Caldwell.
Todd knows Anthony.
Joshua knows Barry.
Hannah knows Ron.
Anthony knows Julie.
Chase knows Todd.
Chan knows Jo.
Ron knows Chan.
Jo knows Felix.
```

Read question:
```text
Which person is linked from Caldwell by the relation 'knows'?
Answer:
```

Target: ' Nigel'; token IDs [65816]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 65816, "token": " Nigel", "logit": 19.0, "probability": 0.48417937755584717}, {"token_id": 29106, "token": " Hugh", "logit": 18.0, "probability": 0.17811962962150574}, {"token_id": 25179, "token": " Philip", "logit": 18.0, "probability": 0.17811962962150574}, {"token_id": 33121, "token": " Neil", "logit": 16.875, "probability": 0.05782698094844818}, {"token_id": 44218, "token": " Seth", "logit": 16.0, "probability": 0.024105871096253395}]

## t2_t1_wording, episode 8111687887098378937, matrix

Support (encoded separately):
```text
Conte knows Allen.
Connor knows Conte.
Mark knows Harm.
Matthias knows Paige.
Marine knows Christopher.
Harm knows Boris.
Boris knows Marine.
Kathy knows Hugo.
Hugo knows Mark.
Christopher knows Bruce.
Carolina knows Kathy.
Paige knows Connor.
```

Read question:
```text
Which person is linked from Conte by the relation 'knows'?
Answer:
```

Target: ' Allen'; token IDs [20060]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 88245, "token": " Patty", "logit": 16.0, "probability": 0.10013832896947861}, {"token_id": 41563, "token": " Catherine", "logit": 15.5625, "probability": 0.0646541640162468}, {"token_id": 20060, "token": " Allen", "logit": 15.5625, "probability": 0.0646541640162468}, {"token_id": 78093, "token": " Judith", "logit": 15.125, "probability": 0.04174386337399483}, {"token_id": 59802, "token": " Autumn", "logit": 15.0625, "probability": 0.039214733988046646}]

## t2_t1_wording, episode 7494498776518826428, matrix

Support (encoded separately):
```text
Barber knows Lynn.
Drake knows Bray.
Sandra knows Corey.
Cord knows Esther.
Chan knows Kristen.
Dale knows Susan.
Esther knows Barber.
Ella knows Dale.
Kristen knows Sandra.
Kata knows Cord.
Corey knows Kata.
Susan knows Drake.
```

Read question:
```text
Which person is linked from Barber by the relation 'knows'?
Answer:
```

Target: ' Lynn'; token IDs [44538]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 45107, "token": " Raven", "logit": 19.25, "probability": 0.7434828877449036}, {"token_id": 67473, "token": " Tiffany", "logit": 16.125, "probability": 0.03266635537147522}, {"token_id": 69217, "token": " Barton", "logit": 16.0, "probability": 0.028827957808971405}, {"token_id": 70371, "token": " Sophia", "logit": 15.25, "probability": 0.013617363758385181}, {"token_id": 65721, "token": " Iris", "logit": 15.125, "probability": 0.012017281726002693}]

## t2_t1_wording, episode 6244493409239580883, matrix

Support (encoded separately):
```text
Julie knows Sydney.
Harm knows Julie.
Hardy knows Otto.
Bowman knows Lore.
Otto knows Olivier.
Christine knows Gala.
Claude knows Christine.
Rosie knows Patty.
Sydney knows Rosie.
Olivier knows Bowman.
Lore knows Chambers.
Gala knows Hardy.
```

Read question:
```text
Which person is linked from Rosie by the relation 'knows'?
Answer:
```

Target: ' Patty'; token IDs [88245]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 88245, "token": " Patty", "logit": 17.75, "probability": 0.7351518273353577}, {"token_id": 98974, "token": " Richie", "logit": 13.6875, "probability": 0.012648986652493477}, {"token_id": 7270, "token": " Sunday", "logit": 13.375, "probability": 0.009254195727407932}, {"token_id": 48567, "token": " Lucky", "logit": 13.3125, "probability": 0.00869351252913475}, {"token_id": 74571, "token": " Cynthia", "logit": 13.25, "probability": 0.008166799321770668}]

## t2_t1_wording, episode 3285038830464699338, matrix

Support (encoded separately):
```text
Mona knows Viktor.
Joel knows Leonard.
Ole knows Gene.
Viktor knows De.
Gandhi knows Ole.
Abdullah knows Scott.
Xavier knows Mona.
Damon knows Castillo.
Lisa knows Damon.
Scott knows Lisa.
Gene knows Joel.
De knows Abdullah.
```

Read question:
```text
Which person is linked from Joel by the relation 'knows'?
Answer:
```

Target: ' Leonard'; token IDs [40854]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 95683, "token": " Castillo", "logit": 16.875, "probability": 0.3150663375854492}, {"token_id": 61577, "token": " Bolton", "logit": 15.5625, "probability": 0.08479894697666168}, {"token_id": 84190, "token": " Buchanan", "logit": 15.25, "probability": 0.062040239572525024}, {"token_id": 63444, "token": " Clayton", "logit": 14.875, "probability": 0.0426395945250988}, {"token_id": 38286, "token": " Bennett", "logit": 14.625, "probability": 0.03320774808526039}]

## t2_t1_wording, episode 1287256782801180009, soft

Support (encoded separately):
```text
Caldwell knows Nigel.
Julie knows Joshua.
Barry knows Shane.
Shane knows Caldwell.
Todd knows Anthony.
Joshua knows Barry.
Hannah knows Ron.
Anthony knows Julie.
Chase knows Todd.
Chan knows Jo.
Ron knows Chan.
Jo knows Felix.
```

Read question:
```text
Which person is linked from Caldwell by the relation 'knows'?
Answer:
```

Target: ' Nigel'; token IDs [65816]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 65816, "token": " Nigel", "logit": 17.625, "probability": 0.2898615598678589}, {"token_id": 25179, "token": " Philip", "logit": 17.125, "probability": 0.17580993473529816}, {"token_id": 33121, "token": " Neil", "logit": 16.625, "probability": 0.10663411021232605}, {"token_id": 44218, "token": " Seth", "logit": 16.125, "probability": 0.06467685103416443}, {"token_id": 29106, "token": " Hugh", "logit": 16.125, "probability": 0.06467685103416443}]

## t2_t1_wording, episode 8111687887098378937, soft

Support (encoded separately):
```text
Conte knows Allen.
Connor knows Conte.
Mark knows Harm.
Matthias knows Paige.
Marine knows Christopher.
Harm knows Boris.
Boris knows Marine.
Kathy knows Hugo.
Hugo knows Mark.
Christopher knows Bruce.
Carolina knows Kathy.
Paige knows Connor.
```

Read question:
```text
Which person is linked from Conte by the relation 'knows'?
Answer:
```

Target: ' Allen'; token IDs [20060]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 11044, "token": " Peter", "logit": 14.625, "probability": 0.06092899292707443}, {"token_id": 7355, "token": " Ben", "logit": 14.25, "probability": 0.041875846683979034}, {"token_id": 61695, "token": " Maggie", "logit": 14.125, "probability": 0.03695530444383621}, {"token_id": 51079, "token": " Jenny", "logit": 14.0, "probability": 0.03261294215917587}, {"token_id": 23133, "token": " Hunter", "logit": 14.0, "probability": 0.03261294215917587}]

## t2_t1_wording, episode 7494498776518826428, soft

Support (encoded separately):
```text
Barber knows Lynn.
Drake knows Bray.
Sandra knows Corey.
Cord knows Esther.
Chan knows Kristen.
Dale knows Susan.
Esther knows Barber.
Ella knows Dale.
Kristen knows Sandra.
Kata knows Cord.
Corey knows Kata.
Susan knows Drake.
```

Read question:
```text
Which person is linked from Barber by the relation 'knows'?
Answer:
```

Target: ' Lynn'; token IDs [44538]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 69697, "token": " Tara", "logit": 14.8125, "probability": 0.06448592990636826}, {"token_id": 31432, "token": " Rachel", "logit": 14.5625, "probability": 0.05022169277071953}, {"token_id": 86618, "token": " Ashton", "logit": 14.375, "probability": 0.04163524508476257}, {"token_id": 29201, "token": " Kate", "logit": 14.375, "probability": 0.04163524508476257}, {"token_id": 26518, "token": " Chelsea", "logit": 14.1875, "probability": 0.0345168337225914}]

## t2_t1_wording, episode 6244493409239580883, soft

Support (encoded separately):
```text
Julie knows Sydney.
Harm knows Julie.
Hardy knows Otto.
Bowman knows Lore.
Otto knows Olivier.
Christine knows Gala.
Claude knows Christine.
Rosie knows Patty.
Sydney knows Rosie.
Olivier knows Bowman.
Lore knows Chambers.
Gala knows Hardy.
```

Read question:
```text
Which person is linked from Rosie by the relation 'knows'?
Answer:
```

Target: ' Patty'; token IDs [88245]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 42360, "token": " Lauren", "logit": 13.5, "probability": 0.04559038579463959}, {"token_id": 22417, "token": " Victoria", "logit": 13.0625, "probability": 0.02943536452949047}, {"token_id": 8120, "token": " Clinton", "logit": 13.0625, "probability": 0.02943536452949047}, {"token_id": 44030, "token": " Nicole", "logit": 12.8125, "probability": 0.022924285382032394}, {"token_id": 26518, "token": " Chelsea", "logit": 12.8125, "probability": 0.022924285382032394}]

## t2_t1_wording, episode 3285038830464699338, soft

Support (encoded separately):
```text
Mona knows Viktor.
Joel knows Leonard.
Ole knows Gene.
Viktor knows De.
Gandhi knows Ole.
Abdullah knows Scott.
Xavier knows Mona.
Damon knows Castillo.
Lisa knows Damon.
Scott knows Lisa.
Gene knows Joel.
De knows Abdullah.
```

Read question:
```text
Which person is linked from Joel by the relation 'knows'?
Answer:
```

Target: ' Leonard'; token IDs [40854]; answer-position index 15 / token ':'.

Top five next tokens: [{"token_id": 72249, "token": " Damon", "logit": 13.875, "probability": 0.04661485552787781}, {"token_id": 85016, "token": " Alec", "logit": 13.5625, "probability": 0.03410416096448898}, {"token_id": 44218, "token": " Seth", "logit": 13.5, "probability": 0.0320378914475441}, {"token_id": 54652, "token": " Corey", "logit": 13.1875, "probability": 0.023439424112439156}, {"token_id": 44457, "token": " Rebecca", "logit": 13.1875, "probability": 0.023439424112439156}]
