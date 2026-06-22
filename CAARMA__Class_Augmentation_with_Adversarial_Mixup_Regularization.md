CAARMA: Class Augmentation with Adversarial Mixup Regularization
MassaBaali,XiangLi,HaoChen,SyedAbdulHannan,RitaSingh,BhikshaRaj
CarnegieMellonUniversity
mbaali@cs.cmu.edu
Abstract
Speakerverificationisatypicalzero-shotlearn-
ingtask,whereinferenceofunseenclassesis
| performed                 | by comparing embeddings | of test           |     |     |     |     |
| ------------------------- | ----------------------- | ----------------- | --- | --- | --- | --- |
| instancestoknownexamples. |                         | Themodelsper-     |     |     |     |     |
| forming                   | inference must hence    | naturally gen-    |     |     |     |     |
| erate embeddings          | that cluster            | same-class in-    |     |     |     |     |
| stancescompactly,         | whilemaintainingsepara- |                   |     |     |     |     |
| tionacrossclasses.        | Inordertolearntodoso,   |                   |     |     |     |     |
| they are                  | typically trained       | on a large number |     |     |     |     |
| of classes                | (speakers), often       | using specialized |     |     |     |     |
losses. Howeverreal-worldspeakerdatasetsof-
tenlacktheclassdiversityneededtoeffectively
learnthisinageneralizablemanner. Weintro- Figure1:(a)Whentrainedwithfewerclassesthemodel
duceCAARMA,aclassaugmentationframe- can spread the embeddings of individual classes out
|     |     |     | while still | learning to classify | the training | data accu- |
| --- | --- | --- | ----------- | -------------------- | ------------ | ---------- |
workthataddressesthisproblembygenerating
synthetic classes through data mixing in the ratelyandwithlargemargins. Thiswillnot,however
translatetocompactrepresentationsfornewerunseen
| embedding | space, expanding | the number of |     |     |     |     |
| --------- | ---------------- | ------------- | --- | --- | --- | --- |
classes. (b)Withadditionalsyntheticclasses(shaded
| trainingclasses. | Toensuretheauthenticityof |     |     |     |     |     |
| ---------------- | ------------------------- | --- | --- | --- | --- | --- |
thesyntheticclassesweadoptanoveladversar- grey), the model must now learn to compact classes
|     |     |     | more. This | will translate | to more compact | unseen |
| --- | --- | --- | ---------- | -------------- | --------------- | ------ |
ialrefinementmechanismthatminimizescate-
| goricaldistinctionsbetweensyntheticandreal |                    |             | classesaswell. |     |     |     |
| ------------------------------------------ | ------------------ | ----------- | -------------- | --- | --- | --- |
| classes.                                   | We evaluate CAARMA | on multiple |                |     |     |     |
Toaddressthechallengeoflimitedclassdiver-
speakerverificationtasks,aswellasotherrep-
resentativezero-shotcomparison-basedspeech sityintrainingdatasets,acommonissueinspeaker
analysis tasks and obtain consistent improve- verification, we propose a novel augmentation-
ments: our framework demonstrates a signif- basedtrainingparadigm. Thisapproachleverages
icant improvement of 8% over all baseline syntheticdataaugmentationtoenhancetherobust-
| models. | The code is available | at: https: |     |     |     |     |
| ------- | --------------------- | ---------- | --- | --- | --- | --- |
nessandgeneralizationcapabilitiesofspeakerver-
//github.com/massabaali7/CAARMA/
ificationsystems,particularlyinlow-diversityen-
|     |     |     | vironments. | Ourmethodnotonlystaystruetothe |     |     |
| --- | --- | --- | ----------- | ------------------------------ | --- | --- |
1 Introduction
essenceofZSLbyfacilitatingeffectivegeneraliza-
Speakerverificationisfundamentallyazero-shot tiontonewspeakersbutalsointroducesapractical
learning(ZSL)task, whereverificationisaccom- solution to overcome the inherent limitations of
plishedbycomparingembeddingsfromenrollment traditionaltrainingdatasets.
andverificationsampleswithouttheneedforfur- Effective zero-shot learning lies in generating
ther training (Wan et al., 2018). This process embeddings that cluster same-class (in our case,
aligns with the principles of ZSL, where models same-speaker)instancescloselywhilemaintaining
areexpectedtooperateeffectivelyonunseendata. separation between different classes (Zhu et al.,
Therefore,whilethefollowingdiscussionisframed 2019). Traditional training approaches for ZSL
withinthebroadercontextofZSL,itisspecifically modelsrelyontwokeycomponents: exposureto
tailoredtoaddressthechallengesinspeakerverifi- a large number of diverse classes and the use of
| cation. |     |     | specializedlossfunctionsthatpromotebothinter- |     |     |     |
| ------- | --- | --- | --------------------------------------------- | --- | --- | --- |

classseparationandintra-classcompactness(Min izationtounseenclasses(Xieetal.,2022).
etal.,2020). Theunderlyingprinciplehereisthat In this paper, we introduce Class Augmen-
by learning from a sufficiently large number of tation with AdversaRial Mixup regulariAztion
classes,andthroughproperencouragementembod- (CAARMA), a data augmentation framework to
ied in the losses, the model learns not merely to introduce synthetic classes (speakers) to enhance
separatetheclassesithasseen,butthemoregen- ZSLtrainingforspeakerverification. CAARMA
eralprinciplethatinstancesfromaclassmustbe utilizesamixup-likestrategytogeneratedatafrom
clustered closely together while begin separated fictitiousspeakers. However,unlikeconventional
fromthosefromotherclasses(Xianetal.,2018). mixupwhichmixesdataintheinputspace,which
However, when the training datasets lack the would arguably be meaningless in our setting (a
necessary variety of classes (speakers), this can straight-forwardmixoftwospeechrecordingswill
severely limit the model’s ability to develop ro- merely result in a mixed signal, and not a new
bust and transferable representations (Xie et al., speaker),themixupisperformedintheembedding
2022). Indeed, it may be argued that in the high- spaceinamannerthatpermitsassignmentofnew
dimensional space of the embeddings, any finite classidentitiestothemixed-updata. Critically,we
set of training classes is insufficient to cover the must now ensure that the mixed-up embeddings
spaceadequately. Thislimitationleadstomodels resemble those from actual speakers. We do so
that fail to generalize effectively to unseen cate- through a discriminator that is used to minimize
gories,resultinginsuboptimalzero-shotinference categoricaldistinctionsbetweensyntheticandau-
performance(Guptaetal.,2021). thenticdatathroughadversarialtraining.
Popular approaches to address training data
We demonstrate CAARMA’s effectiveness
limitations often rely on data augmentation tech-
through extensive evaluation on speaker verifica-
niques. Methods such as AutoAugment (Cubuk
tion,whereitachievessubstantialimprovementsin
etal.,2019)andSpecAugment(Parketal.,2019)
generalizingtodiversespeakerdistributions. Ad-
generatenewsamplesbymodifyingexistingones
ditional experiments on other ZSL speech tasks
throughtransformationslikegeometricdistortions,
furthervalidateourapproach’sbroadapplicability.
time warping, and frequency masking. However,
Ourmaincontributionsareasfollows:
while these techniques increase intra-class diver-
sity, they do not introduce new classes, limiting
theireffectivenessinzero-shotlearningscenarios. • We introduce CAARMA, a novel class aug-
Of most relevance to our paper are mixup-based mentationframeworkthataddressesthefun-
data augmentation techniques, e.g. (Verma et al., damentallimitationofclassdiversityinzero-
2019;Yunetal.,2019),thataimtoenhancetraining shotlearningbygeneratingsyntheticclasses
bygeneratingnewsamplesthroughinterpolation termedasSytheticLabelMixup(SL-Mixup)
(Han et al., 2021) of both the features and their throughembedding-spacemixing,ratherthan
labels. Regardless of the interpolation, yet these conventionalinput-spaceaugmentation.
methods also do not generate new classes; they
merelyimprovethegeneralizationofthemodelby
• Wedevelopanadversarialtrainingmechanism
mapping mixed data to mixed-class labels. Still
that ensures the synthetic classes generated
othermethodsusegenerativemodelssuchasVari-
through our mixing strategy maintain statis-
ational Autoencoders (VAEs), Generative Adver-
tical authenticity by minimizing categorical
sarialNetworks(GANs),diffusionmodelsetc. to
distinctions between real and synthetic em-
generate authentically novel data to enhance the
beddings.
training(Minetal.,2019);howeverthesetooare
generallyrestrictedtogeneratingnovelinstances
for known classes, limiting their effectiveness in • Weachievesignificantperformanceimprove-
zero-shotscenarios(Pourpanahetal.,2022). Thus, ments in zero-shot inference, demonstrated
whiletheseapproachesaregenerallyverysuccess- through an 8% improvement over baseline
ful in improving generalization in classification modelsinspeakerverificationtasks,withen-
problems,theyfailataddressingtheproblemZSL hancedgeneralizationtodiversespeakerdis-
learning faces, that of increasing the number of tributionsandverifiedapplicabilityacrossvar-
classesthemselves,leadingtoinconsistentgeneral- iouszero-shotlearningtasks.

2 Related-Work
|     |     |     |     |     |     |     | els. A | notable | innovation | is  | ConversaSynth, |     | a   |
| --- | --- | --- | --- | --- | --- | --- | ------ | ------- | ---------- | --- | -------------- | --- | --- |
frameworkutilizinglargelanguagemodels(LLMs)
| Mixup. | Thedevelopmentofmixupstrategieshas |     |     |     |     |     |     |     |     |     |     |     |     |
| ------ | ---------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
togeneratesyntheticconversationalaudioacross
evolvedsubstantiallysinceMixup’soriginalintro-
|                             |     |     |     |                |     |     | varied persona |     | settings | (Kyaw | and Chan, | 2024). |     |
| --------------------------- | --- | --- | --- | -------------- | --- | --- | -------------- | --- | -------- | ----- | --------- | ------ | --- |
| ductionby(Zhangetal.,2018), |     |     |     | whichgenerated |     |     |                |     |          |       |           |        |     |
Thismethodbeginswithgeneratingtext-baseddi-
virtualsamplesandmixedlabelsbylinearlycom-
|     |     |     |     |     |     |     | alogues, | which | are then | rendered | into | audio | us- |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----- | -------- | -------- | ---- | ----- | --- |
biningtwoinputsamplesandtheircorresponding
|     |     |     |     |     |     |     | ing text-to-speech |     | (TTS) | systems. | The | synthetic |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | ----- | -------- | --- | --------- | --- |
labels. Thispioneeringmethodprovedparticularly
|     |     |     |     |     |     |     | datasets | produced | are noted | for | their realism |     | and |
| --- | --- | --- | --- | --- | --- | --- | -------- | -------- | --------- | --- | ------------- | --- | --- |
successfulinenhancingdatadiversityandimprov-
|                                               |     |     |     |     |     |     | topicvariety,  |     | provingbeneficialfortaskssuchas |     |                   |     |     |
| --------------------------------------------- | --- | --- | --- | --- | --- | --- | -------------- | --- | ------------------------------- | --- | ----------------- | --- | --- |
| inggeneralizationinvisualclassificationtasks. |     |     |     |     |     | Ex- |                |     |                                 |     |                   |     |     |
|                                               |     |     |     |     |     |     | audio tagging, |     | classification,                 |     | and multi-speaker |     |     |
tensionssuchasManifoldMix(Vermaetal.,2019)
|          |              |       |            |         |             |      | speechrecognition. |     | ThesecapabilitiesmakeCon- |     |            |         |     |
| -------- | ------------ | ----- | ---------- | ------- | ----------- | ---- | ------------------ | --- | ------------------------- | --- | ---------- | ------- | --- |
| applied  | this concept | to    | hidden     | layers, | while       | Cut- |                    |     |                           |     |            |         |     |
|          |              |       |            |         |             |      | versaSynth         | a   | valuable tool             | for | developing | robust, |     |
| Mix (Yun | et al.,      | 2019) | introduced | a       | patch-based |      |                    |     |                           |     |            |         |     |
adaptableAImodelsthatcanhandlediverseaudio
| approach       | by blending                      |       | rectangular | sections |            | of im- |                                       |     |     |     |                 |       |     |
| -------------- | -------------------------------- | ----- | ----------- | -------- | ---------- | ------ | ------------------------------------- | --- | --- | --- | --------------- | ----- | --- |
|                |                                  |       |             |          |            |        | dataandcomplexconversationalcontexts. |     |     |     |                 | Inthe |     |
| ages, offering | a                                | novel | alternative | for      | augmenting |        |                                       |     |     |     |                 |       |     |
|                |                                  |       |             |          |            |        | domainofspeakerverification,          |     |     |     | SpeechMixintro- |       |     |
| trainingdata.  | Subsequentmixupstrategiesfocused |       |             |          |            |        |                                       |     |     |     |                 |       |     |
ducesanovelmethodbymixingspeechatthewave-
ontailoringdatamixingtospecificcontextsorim-
formlevel,carefullyadjustingratiostopreservethe
| proving | the precision |     | of mixing. | Static | policies |     |     |     |     |     |     |     |     |
| ------- | ------------- | --- | ---------- | ------ | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
distinctcharacteristicsofspeakeridentity(Jindal
likeSmoothMix(Leeetal.,2020),GridMix(Baek
|     |     |     |     |     |     |     | etal.,2020). | However,likemanyothergenerative |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ------------------------------- | --- | --- | --- | --- | --- |
etal.,2021),andResizeMix(Qinetal.,2020)used
andaugmentationtechniques,SpeechMixprimar-
| hand-crafted | cutting | techniques, |     | while | dynamic |     |     |     |     |     |     |     |     |
| ------------ | ------- | ----------- | --- | ----- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
ilyfocusesonmanipulatingknownspeakervoices
approachessuchasPuzzleMix(Kimetal.,2020)
ratherthangeneratingnewidentities,therebylimit-
| and AlignMix | (Venkataramanan |     |     | et  | al., 2022) | in- |     |     |     |     |     |     |     |
| ------------ | --------------- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
ingitsutilityforenhancingspeakerdiversitycrit-
corporatedoptimal-transportmethodstodetermine
|             |      |         |              |     |     |        | ical for | effective | zero-shot | learning. |     | In the | con- |
| ----------- | ---- | ------- | ------------ | --- | --- | ------ | -------- | --------- | --------- | --------- | --- | ------ | ---- |
| mix regions | with | greater | flexibility. |     | For | Vision |          |           |           |           |     |        |      |
textofspeakerverification,discriminativeneural
Transformers,strategiessuchasTransMix(Chen
clustering(Lietal.,2021)employsclusteringtech-
| et al., 2022) | and | TokenMix | (Liu | et  | al., 2022) | fo- |     |     |     |     |     |     |     |
| ------------- | --- | -------- | ---- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
niquestoenhancespeakerdiarizationbylearning
| cused on | leveraging | attention |     | mechanisms |     | to re- |     |     |     |     |     |     |     |
| -------- | ---------- | --------- | --- | ---------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
discriminativeembeddings,butitdoesnotaddress
finemixingoperations,particularlyfortransformer
classdiversitythroughsyntheticclassgeneration
| architectures                               | (Dosovitskiy |         | et    | al., 2021). |            | Recent |           |     |                     |     |         |           |     |
| ------------------------------------------- | ------------ | ------- | ----- | ----------- | ---------- | ------ | --------- | --- | ------------------- | --- | ------- | --------- | --- |
|                                             |              |         |       |             |            |        | as CAARMA |     | does. Synthio       |     | employs | a unique  |     |
| developments                                | have         | adapted | mixup |             | techniques | to     |           |     |                     |     |         |           |     |
|                                             |              |         |       |             |            |        | approach  | by  | using text-to-audio |     | (T2A)   | diffusion |     |
| tasksbeyondclassification,suchasregression. |              |         |       |             |            | C-     |           |     |                     |     |         |           |     |
modelstoaugmentsmall-scaleaudioclassification
Mixup,forinstance,appliessamplemixingbased
|          |            |       |             |     |          |     | datasets(Ghoshetal.,2026). |     |     |     | Itenhancescomposi- |     |     |
| -------- | ---------- | ----- | ----------- | --- | -------- | --- | -------------------------- | --- | --- | --- | ------------------ | --- | --- |
| on label | distances, | using | a symmetric |     | Gaussian |     |                            |     |     |     |                    |     |     |
tionaldiversityandmaintainsacousticconsistency
| kernel | to select | samples | that | improve | regression |     |     |     |     |     |     |     |     |
| ------ | --------- | ------- | ---- | ------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
byaligningT2A-generatedsyntheticsampleswith
| performance(Yaoetal.,2022). |     |     |     | Furtherenhancing |     |     |     |     |     |     |     |     |     |
| --------------------------- | --- | --- | --- | ---------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
theoriginaldatasetusingpreferenceoptimization.
| robustness, | RC-Mixup |     | integrates | C-Mixup |     | with |     |     |     |     |     |     |     |
| ----------- | -------- | --- | ---------- | ------- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
Additionally,exploringstyletransferinsynthetic
| multi-round | robust  | training, |                | creating | a feedback |       |                             |     |     |     |               |     |     |
| ----------- | ------- | --------- | -------------- | -------- | ---------- | ----- | --------------------------- | --- | --- | --- | ------------- | --- | --- |
|             |         |           |                |          |            |       | audio,recentworkbyUedaetal. |     |     |     | employsaVITS- |     |     |
| loop where  | C-Mixup |           | helps identify |          | cleaner    | data, |                             |     |     |     |               |     |     |
basedvoiceconversionmodel,conditionedonthe
| and robust | training | improves |     | the quality |     | of data |     |     |     |     |     |     |     |
| ---------- | -------- | -------- | --- | ----------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
fundamentalfrequency(F0),toproduceexpressive
| for mixing | (Hwang | et  | al., 2024). | These |     | special- |            |      |         |         |        |      |      |
| ---------- | ------ | --- | ----------- | ----- | --- | -------- | ---------- | ---- | ------- | ------- | ------ | ---- | ---- |
|            |        |     |             |       |     |          | variations | from | neutral | speaker | voices | (?). | This |
izedapproachesrevealmixup’sadaptabilityacross
|     |     |     |     |     |     |     | methodachievescross-speakerstyletransferin |     |     |     |     |     | a   |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------------------ | --- | --- | --- | --- | --- | --- |
variousmachinelearningtasks,enhancingmodel
FastPitch-basedTTSsystem,incorporatingastyle
| performanceanddataresilience. |     |     |     | However,it’sim- |     |     |         |             |     |                  |     |      |     |
| ----------------------------- | --- | --- | --- | --------------- | --- | --- | ------- | ----------- | --- | ---------------- | --- | ---- | --- |
|                               |     |     |     |                 |     |     | encoder | pre-trained | on  | timbre-perturbed |     | data | to  |
portanttonotethatnoneofthesemethodsinvolve
|     |     |     |     |     |     |     | preventspeakerleakage. |     |     | Thistechniqueenhances |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---------------------- | --- | --- | --------------------- | --- | --- | --- |
mixingintheembeddingspace,whichcouldallow
theutilityofsyntheticdatainapplicationsrequir-
forthecreationofentirelynewandsyntheticclass
|     |     |     |     |     |     |     | ing rich | stylistic | diversity. | These | developments |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | --------- | ---------- | ----- | ------------ | --- | --- |
identities,
underlinetheincreasingsophisticationofsynthetic
Syntheticspeech. Recentadvancementsinsyn- audio generation techniques, from multi-speaker
thetic audio generation have emphasized the cre- conversations in ConversaSynth to Synthio’s op-
ationofdiverseandhigh-qualitydatasets,crucial timizedT2Aaugmentationforclassification,and
for training and evaluating audio-based AI mod- cross-speakerstyletransferswithVITS-basedmod-

els. They collectively demonstrate the power of theclasslabelsintheloss. Inaddition,themodel
synthetic data to enrich audio model training by alsoattemptstoadversariallyfoolthediscriminator.
addingdiversityandrealism. However,thesemeth- Oncethemodelistrained,thediscriminatorisno
odsstillfacelimitationsingeneratingentirelynew longerneededandisdiscarded.
speakeridentities,whichiscriticalforexpanding
3.2 Framework
therangeofrecognizablevoicesinspeakerverifi-
cationsystems. CAARMAaddressesthisgapby Ourframeworkconsistsofthreemaincomponents:
directlymixingintheembeddingspace, creating anencoderforembeddinggeneration,asynthetic
syntheticspeakersthatenhancezero-shotlearning label mixup mechanism for class augmentation,
capabilities. CAARMAnotonlypreservesspeaker andanadversarialtrainingschemewithaseman-
characteristics but also significantly expands the ticdiscriminator. Figure2illustratesthecomplete
diversityofspeakeridentities,offeringasuperior
|     |     |     |     |     |     | pipelineofourapproach. |     |     | Theprocessbeginswith |     |     |
| --- | --- | --- | --- | --- | --- | ---------------------- | --- | --- | -------------------- | --- | --- |
solution for training more robust and adaptable a waveform input that is transformed into a Mel-
speakerverificationsystems. spectrogram. This spectrogram serves as input
|     |     |     |     |     |     | to the encoder | E,  | which | generates | embeddings | e   |
| --- | --- | --- | --- | --- | --- | -------------- | --- | ----- | --------- | ---------- | --- |
3 ClassAugmentationwithAdversarial thatcapturediscriminativespeakercharacteristics.
MixupRegularization TheseembeddingsundergoourSL-Mixupstrategy,
|              |     |     |     |     |     | whichgeneratessyntheticembeddingse |     |     |     | bymix- |     |
| ------------ | --- | --- | --- | --- | --- | ---------------------------------- | --- | --- | --- | ------ | --- |
| 3.1 Overview |     |     |     |     |     |                                    |     |     |     | syn    |     |
ingembeddingsebasedontheirclosestneighbor
As mentioned in Section 1, ZSL models learn weights W. Each synthetic embedding receives
their ability to compactly cluster same-class em- a corresponding synthetic label ID within the
syn
| beddings | while | maintaining |     | separation | between |             |                               |     |     |     |     |
| -------- | ----- | ----------- | --- | ---------- | ------- | ----------- | ----------------------------- | --- | --- | --- | --- |
|          |       |             |     |            |         | mini-batch. | Theframeworkemploystwoprimary |     |     |     |     |
classesprimarilythroughexposuretoalargenum- loss functions: the encoder loss L for original
real
ber of classes during training; the more classes embeddings and the synthetic loss L for syn-
syn
they are exposed to in training, the better they thetic embeddings. A Self-Supervised Learning
| are able | to generalize |     | to unseen | classes. | In the |             |        |     |           |               |     |
| -------- | ------------- | --- | --------- | -------- | ------ | ----------- | ------ | --- | --------- | ------------- | --- |
|          |               |     |           |          |        | (SSL) model | serves | as  | the mixup | discriminator |     |
speakerverificationsetting,thistranslatestotrain- todistinguish betweenreal(R)and synthetic(S)
ing the model with recordings from a large num- embeddings. Adiscriminator loss L guidesthe
D
| ber of speakers; |     | the more | the | number | of training |     |     |     |     |     |     |
| ---------------- | --- | -------- | --- | ------ | ----------- | --- | --- | --- | --- | --- | --- |
discriminatortomaximallydistinguishbetweenR
speakersthebetterthemodelgeneralizes. Toim- and S. On the other hand, a generator loss L
gen
prove this generalization CAARMA attempts to guidestheencoderto“fool”thediscriminator,so
increase the number of speakers by creating syn- that it perceives no distinction between real and
| theticspeakerswhiletraining. |     |     |     | Syntheticspeakers |     |     |     |     |     |     |     |
| ---------------------------- | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- | --- |
syntheticembeddings.
maybecreatedthroughgenerativemethodssuch
3.3 Encoder
| as (Cornell | et       | al., 2024); | however   |     | this approach |                     |     |            |     |                  |     |
| ----------- | -------- | ----------- | --------- | --- | ------------- | ------------------- | --- | ---------- | --- | ---------------- | --- |
| does not    | scale.   | Instead,    | CAARMA    |     | creates them  |                     |     |            |     |                  |     |
|             |          |             |           |     |               | The encoder         | E   | transforms |     | Mel-spectrograms |     |
| through     | a simple | mixup       | strategy, | as  | convex com-   |                     |     |            |     | e                |     |
|             |          |             |           |     |               | into discriminative |     | embeddings |     | that capture     |     |
binations of real speakers, with a key distinction: speaker-specific acoustic features. We employ
embedding
the mixup is performed in the space, an MFA-Conformer model (Zhang et al., 2022)
| where the | classes | are | expected | to form | compact |                |     |               |     |                |     |
| --------- | ------- | --- | -------- | ------- | ------- | -------------- | --- | ------------- | --- | -------------- | --- |
|           |         |     |          |         |         | as our encoder |     | architecture, |     | which combines |     |
(andgenerallyconvex)clusters. Inordertoensure feed-forward networks (FFNs), multihead self-
thatthesesyntheticspeakersareindeedrepresenta- attention(MHSA),andconvolutionmodules. The
tiveofactualspeakers,CAARMAutilizesamixup
modelincorporatespositionalembeddingstohan-
discriminator, a discriminator which attempts to dlevariable-lengthinputsequenceseffectively. For
distinguishbetweensyntheticandrealspeakers: if training, we utilize the AM-Softmax function as
thisdiscriminatorisfooled,thesyntheticspeakers
|     |     |     |     |     |     | ourencoderlossL |     | .   |     |     |     |
| --- | --- | --- | --- | --- | --- | --------------- | --- | --- | --- | --- | --- |
real
arestatisticallyindistinguishablefromrealones.
es·(cos(θy)−m)
| When | training | the | model, | a conventional | loss |     |     |     |     |     |     |
| ---- | -------- | --- | ------ | -------------- | ---- | --- | --- | --- | --- | --- | --- |
L = −log
|         |     |          |        |         |      |     | real |     | (cid:80)C |            |     |
| ------- | --- | -------- | ------ | ------- | ---- | --- | ---- | --- | --------- | ---------- | --- |
| such as | the | Additive | Margin | Softmax | (AM- |     |      |     |           | es·cos(θj) |     |
j=1
| Softmax)(Wangetal.,2018)isused. |     |     |     |     | Thesynthetic |     |     |     |     |     |     |
| ------------------------------- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- | --- |
classes,whicharecreateddynamicallyduringtrain- wheresisascalingfactorusedtostabilizegradi-
ing, are included by dynamically also expanding ents, C represents the number of classes, cos(θ )
y

|     |     |     |     |     |     |     |     |     | ℒ   | Speaker 1 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --------- | --- | --- |
$%&'
|              |     |     | ℇ   |             |     |        |     | ℋ      |     |           |        |     |
| ------------ | --- | --- | --- | ----------- | --- | ------ | --- | ------ | --- | --------- | ------ | --- |
| Mel-Spectrum |     |     |     | Real Embed. |     |        |     |        |     | Speaker 2 |        |     |
|              |     |     |     |             |     | Mix-Up |     | Shared |     |           | Mix-Up |     |
ℒ
!"#
ℋ
Speaker Embed.
Syn. Speaker
Syn. Embed.
| ℇ   | Encoder   |     |     |                      |     |     |     |     |     |     |           |     |
| --- | --------- | --- | --- | -------------------- | --- | --- | --- | --- | --- | --- | --------- | --- |
|     | Cls. Head |     |     | Pretrained SSL Model |     |     |     |     |     |     | Real/Fake |     |
ℋ
Multi-layer Features
Figure2: IllustrationofCAARMAframework. (a)Theencoder(E)extractsembeddingsfromMel-spectrograms,
which are processed by a classification head (H) for speaker identification and through Mix-Up for synthetic
embeddinggeneration. (b)BothrealandsyntheticembeddingsarefedintoapretrainedSSLmodelthatactsasa
discriminator,distinguishingbetweenrealandsyntheticsamples.
denotesthecosinesimilarityforthetrueclass,and Algorithm1SL-Mixup
misanadditivemarginthatenhancesclasssepara-
|     |     |     |     |     |     |     | Input: | Feature | matrix |     | X, Label vector | Y,  |
| --- | --- | --- | --- | --- | --- | --- | ------ | ------- | ------ | --- | --------------- | --- |
tionbyincreasinginter-classdistances.
WeightmatrixW
|     |     |     |     |     |     |     | InitializeW |     | ← 0,Y |     | ← 0,X ← 0 |     |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ----- | --- | --------- | --- |
|     |     |     |     |     |     |     |             |     | syn   | syn | syn       |     |
3.4 SyntheticLabelMixup
|     |     |     |     |     |     |     | fory | ∈ Y | do  |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
i
Our SL-Mixup strategy generates synthetic em- distances ← ∥W[:,i] − W[:,j]∥ ∀j ∈
2
label_set\{i}
| beddings | e syn | within each | mini-batch |     | by mixing |     |     |     |     |     |     |     |
| -------- | ----- | ----------- | ---------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
embeddings e according to their closest neigh- neighbor(y ) ← argmin(distances)
i
| bor weights | W,      | as detailed | in Algorithm |     | 1.     | This | endfor |           |     |     |     |     |
| ----------- | ------- | ----------- | ------------ | --- | ------ | ---- | ------ | --------- | --- | --- | --- | --- |
| approach    | ensures | synthetic   | embeddings   |     | remain |      | fori   | ∈ Batchdo |     |     |     |     |
within the same manifold as real embeddings, l ← Y[i], l ← neighbor(l )
|     |     |     |     |     |     |     | 1   |     | 2   |     | 1   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
avoidingarbitrarygeneration. Thestrategycreates W [:,i] ← 0.5×(W[:,l ]+W[:,l ])
|     |     |     |     |     |     |     |     | syn |     |     | 1 2 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
synthetic labels ID and embeddings dynami- Y [i] ← new_label(l ,l )
|                                                |     | syn |     |     |     |     | syn      |     |          | 1   | 2   |     |
| ---------------------------------------------- | --- | --- | --- | --- | --- | --- | -------- | --- | -------- | --- | --- | --- |
|                                                |     |     |     |     |     |     | index[i] |     | ← find(Y | = l | )   |     |
| callyduringtraining,enablingeffectiverepresen- |     |     |     |     |     |     |          |     |          | 2   |     |     |
tationlearningandfacilitatingthepotentialuseof X [i] ← 0.5×(X[i]+X[index[i],:])
syn
| unlabeleddata. |     | Toensurethatsyntheticspeakers |     |     |     |     | endfor |     |     |     |     |     |
| -------------- | --- | ----------------------------- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- |
are minimally confusable with their component Return: X syn ,Y syn ,W syn
speakers,weonlycombinepairsofspeakerswitha
| fixedweightof0.5. |     | Thisapproachmaintainsabal- |     |     |     |     |     |     |     |     |     |     |
| ----------------- | --- | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ancedcontributionfromeachcomponentspeaker, integrationensuresproperalignmentofsynthetic
embeddingswithintheembeddingmanifold.
preventingsyntheticembeddingsfromcollapsing
intoasingleidentitywhilemaintaininginter-class
|             |     |           |      |      |          |     | 3.5 AdversarialTraining |     |     |     |     |     |
| ----------- | --- | --------- | ---- | ---- | -------- | --- | ----------------------- | --- | --- | --- | --- | --- |
| separation. | The | synthetic | loss | L is | computed |     |                         |     |     |     |     |     |
syn
using the AM-Softmax loss function applied to Our adversarial training process alternates be-
syntheticembeddingse . Thislossisintegrated tween optimizing the encoder and discriminator,
syn
into the main encoder loss L , scaled by 1/λ, as described in Algorithm 2. This optimization
real
where λ represents the number of speakers. This schemecontinuouslyrefinestheembeddingmani-

Algorithm2AdversarialTrainingwithSynthetic representationstoproviderichergradientsduring
Embeddings adversarial training, improving the stability and
Input: Featureextractorf(X),ModelM,Dis- quality of the learned embeddings. The seman-
criminatorD,DatasetD (waveformsX andla- tic discriminator processes embeddings through
belsY),Adversarialweightλ an adapter module that projects them into a com-
adv
foreachepoche ∈ [1,N ]do patible feature space. The adapter consists of a
epochs
foreachbatch(X,Y) ∈ D do down-projectionlayerwithspectralnormalization,
ExtractfeaturesF = Mel(X) followed by fully connected layers with GELU
Computeembeddingse = E(F) activation(HendrycksandGimpel,2016). Weem-
ComputeAM-SoftmaxlossL ploy skip connections and layer normalization to
real
Generate synthetic embeddings e via ensurestabletraining. Thediscriminatorextracts
syn
mixup featuresfrommultipleHuBERT(Hsuetal.,2021)
Compute real predictions D(e) and fake layers(7,9,11,and12)tocapturediversespeaker
predictionsD(e ) characteristics. Thesefeaturesarecombinedusing
syn
Computediscriminatorloss: learnable weights and processed through a resid-
L = BCE(D(e),1)+BCE(D(e ),0) ualclassificationblockwithspectralnormalization
D syn
UpdateD using∇L andLeakyReLUactivation(Xu,2015)todetermine
D
Computegeneratorloss whetheranembeddingisrealorsynthetic.
L = BCE(D(e ),1)+BCE(D(e),0)
G syn 4 Experiments
Adjustλ basedonL /L
adv real G
ComputetotallossL = L +λ L 4.1 Datasets
total real adv G
We utilize four datasets in our proposed ap-
UpdateM using∇L
total proach: VoxCeleb1 (Nagrani et al., 2017), Vox-
endfor
Celeb2 (Chung et al., 2018), and two datasets
endfor
fromtheDynamic-SUPERB(Huangetal.,2024)
Return: TrainedM andD
benchmarksuchasHowFarAreYouandDailyTalk
datasets. Thedatasetstatisticsaresummarizedin
Table1. Thesedatasetswereemployedacrossdif-
foldthroughtheinteractionbetweenrealandsyn-
ferenttaskstoevaluatetheadaptabilityandgener-
theticembeddings:
alizabilityofourpipeline:
• DiscriminatorTraining: Thediscriminator
• SpeakerIdentification: Theprimarytaskof
D learnstodifferentiatebetweenrealembed-
ourstudyinvolvesspeakeridentificationusing
dingseandsyntheticembeddingse using
syn
VoxCeleb1andVoxCeleb2. Theselarge-scale
featuresextractedfrommultiplemodellayers.
datasetscontainspeechrecordingsfromthou-
Thediscriminatorlossisdefinedas:
sandsofspeakers.
L = BCE(D(e),1)+BCE(D(e ),0)
D syn
• Speaker Distance Estimation The How-
(1)
FarAreYou dataset originates from the
where BCE represents binary cross-entropy
3DSpeakerdataset,designedtoassessthedis-
loss.
tanceofaspeakerfromtherecordingdevice.
• Generator Loss: The encoder incorporates The task involves predicting distance labels
a generator loss L that guides embedding (e.g.,0.4m,2.0m)basedonspeechrecordings.
G
alignmentwiththemanifoldstructure:
• EmotionRecognition: WeusetheDailyTalk
L = BCE(D(e ),1)+BCE(D(e),0) datasettoclassifytheemotionalstate(anger,
G syn
(2) disgust, fear, happiness, sadness, surprise,
neutral) of a speaker based on speech utter-
3.6 MixupDiscriminator
ances. Thisdatasetcontainsspeechsamples
Toenhancethediscriminativepowerofourframe- labeledwithsevendistinctemotioncategories.
work,weincorporateaself-supervisedmodel(Hu- To maintain consistency with other datasets,
BERT) (Hsu et al., 2021) as a mixup discrimina- weresampleallrecordingsto16kHzbefore
tor. This discriminator leverages the pre-trained processing.

| ID  | DATASET |     | CLASSES |     | UTTERANCES | ENCODER |     | BASELINE |     | AT  | AT+L |     |
| --- | ------- | --- | ------- | --- | ---------- | ------- | --- | -------- | --- | --- | ---- | --- |
syn
|     |              |     |      |     |          | ECAPATDNN    |     | 4.22 |     | 3.96 | 3.87 |     |
| --- | ------------ | --- | ---- | --- | -------- | ------------ | --- | ---- | --- | ---- | ---- | --- |
| 1   | VOXCELEB1    |     | 1211 |     | 153,516  |              |     |      |     |      |      |     |
|     |              |     |      |     |          | MFACONFORMER |     | 3.33 |     | 3.18 | 3.09 |     |
| 2   | VOXCELEB2    |     | 5994 |     | 1,087135 |              |     |      |     |      |      |     |
| 3   | HOWFARAREYOU |     |      | 3   | 3,000    |              |     |      |     |      |      |     |
4 DAILYTALK 7 16,600 Table 3: EER Results for two different encoders
ECAPA-TDNNandMFAConformershowingperfor-
Table1: Datasetstatisticsusedinourexperiments. manceinbaseline,AdeversarialTraining(AT),andSyn-
|     |     |     |       |         |     | theticLoss(L | syn | )onVoxCeleb1-O. |     |     |     |     |
| --- | --- | --- | ----- | ------- | --- | ------------ | --- | --------------- | --- | --- | --- | --- |
|     | ID  | L   | AT MD | RESULTS |     |              |     |                 |     |     |     |     |
syn
|     | 1   |     |     |     | 3.33 |     |              |        |        |      |        |     |
| --- | --- | --- | --- | --- | ---- | --- | ------------ | ------ | ------ | ---- | ------ | --- |
|     |     |     |     |     |      | ID  | HIDDENLAYERS |        | EER(%) |      | MINDCF |     |
|     | 2   | ✓   |     |     | 3.28 |     |              |        |        |      |        |     |
|     |     |     |     |     |      | 1   | h ,h         | ,h ,h  |        | 3.22 | 0.31   |     |
|     | 3   |     | ✓   |     | 3.15 |     | 3            | 6 9 12 |        |      |        |     |
|     |     |     |     |     |      | 2   | h ,h         | ,h ,h  |        | 3.12 | 0.30   |     |
|     | 4   |     | ✓ ✓ |     | 3.18 |     | 6            | 7 8 9  |        |      |        |     |
|     |     |     |     |     |      | 3   | h ,h         | ,h ,h  |        | 3.09 | 0.28   |     |
|     | 5   | ✓   | ✓   |     | 3.17 |     | 7 9          | 11 12  |        |      |        |     |
|     | 6   | ✓   | ✓ ✓ |     | 3.09 |     |              |        |        |      |        |     |
Table4: Ablationstudyofdifferenthiddenlayersfor
Table 2: EER Results for MFA Conformer baseline, Mixup Discriminator (MD) reporting EER (%) and
minDCF.
AdversarialTraining(AT),MixupDiscriminator(MD),
| andSyntheticLossL |     |     | usingVoxCeleb1fortheSV |     |     |     |     |     |     |     |     |     |
| ----------------- | --- | --- | ---------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
syn
tasks.
|                                           |     |     |     |     |     | including         | {h          | ,h ,h | ,h },    | {h    | ,h ,h  | ,h }, |
| ----------------------------------------- | --- | --- | --- | --- | --- | ----------------- | ----------- | ----- | -------- | ----- | ------ | ----- |
|                                           |     |     |     |     |     |                   | 3           | 6     | 9 12     |       | 7 9    | 11 12 |
|                                           |     |     |     |     |     | {h 6 ,h 7         | ,h 8 ,h 9 } | and   | evaluate | their | impact | on    |
| Table1providesanoverviewofthedatasetsused |     |     |     |     |     | modelperformance. |             |       |          |       |        |       |
inourexperiments,includingthenumberofclasses
| andtotalutterancesperdataset. |     |     |     |     |     | 4.2.3        | ImplementationDetails |              |         |     |     |           |
| ----------------------------- | --- | --- | --- | --- | --- | ------------ | --------------------- | ------------ | ------- | --- | --- | --------- |
|                               |     |     |     |     |     | We implement |                       | all baseline | systems |     | and | discrimi- |
4.2 ExperimentalSetup
|     |     |     |     |     |     | nators | using the | PyTorch | framework |     | (Yun | et al., |
| --- | --- | --- | --- | --- | --- | ------ | --------- | ------- | --------- | --- | ---- | ------- |
4.2.1 ModelArchitecture 2019). Each utterance is randomly segmented
Wetraintwobaselinearchitecturesforspeakerver- into fixed 3-second chunks, with 80-dimensional
|     |     |     |     |     |     | Fbanks | as input | features, | computed |     | using | a 25 |
| --- | --- | --- | --- | --- | --- | ------ | -------- | --------- | -------- | --- | ----- | ---- |
ification:
mswindowlengthanda10msframeshift,with-
| ECAPA-TDNN |     |     | (Desplanques |     | et al., 2020): |     |     |     |     |     |     |     |
| ---------- | --- | --- | ------------ | --- | -------------- | --- | --- | --- | --- | --- | --- | --- |
ContainsthreeSE-Res2Blockswith1024channels outapplyingvoiceactivitydetection. Allmodels
(20.8Mparameters). are trained using AM-Softmax loss with a mar-
|               |     |     |        |         |            | gin of 0.2 | and | a scaling | factor | of  | 30. We | use the |
| ------------- | --- | --- | ------ | ------- | ---------- | ---------- | --- | --------- | ------ | --- | ------ | ------- |
| MFA-Conformer |     |     | (Zhang | et al., | 2022): Em- |            |     |           |        |     |        |         |
AdamWoptimizerwithaninitiallearningrateof
| ploys | 6 Conformer | blocks | with | 256-dimensional |     |     |     |     |     |     |     |     |
| ----- | ----------- | ------ | ---- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- |
encoders,4attentionheads,andconvolutionkernel 0.001formodeltraining,whilethediscriminator
|     |     |     |     |     |     | is optimized | separately |     | with | AdamW | at  | an ini- |
| --- | --- | --- | --- | --- | --- | ------------ | ---------- | --- | ---- | ----- | --- | ------- |
sizeof15(19.7M-20.5Mparameters).
|     |     |     |     |     |     | tial learning | rate | of 2e-4. | To  | prevent | overfitting, |     |
| --- | --- | --- | --- | --- | --- | ------------- | ---- | -------- | --- | ------- | ------------ | --- |
Botharchitecturesgenerate192-dimensionalem-
beddingsforfaircomparison. Foremotionanddis- weapplyaweightdecayof1e-7andusealinear
warmupforthefirst2ksteps,thoughnowarmupis
tancetasks,weutilizeHuBERT-Large(pretrained
|                 |     |      |                  |     |        | appliedtothediscriminator. |      |      |      | Trainingisconducted |      |        |
| --------------- | --- | ---- | ---------------- | --- | ------ | -------------------------- | ---- | ---- | ---- | ------------------- | ---- | ------ |
| on LibriSpeech) |     | with | 1024-dimensional |     | embed- |                            |      |      |      |                     |      |        |
|                 |     |      |                  |     |        | on NVIDIA                  | V100 | GPUs | with | a batch             | size | of 50, |
dingsand768hiddenunits.
andallmodelsaretrainedfor30epochs.
4.2.2 AdversarialTraining
|     |     |     |     |     |     | ComputationalComplexity: |     |     |     | CAARMAintro- |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------ | --- | --- | --- | ------------ | --- | --- |
Weincorporateadversarialtrainingintoourbase-
ducesnoadditionalcomputationalcostduringin-
| lineexperiments. |     | Inthisapproach,eachmodelis |     |     |     |     |     |     |     |     |     |     |
| ---------------- | --- | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ference,asthediscriminatorisnotused,andinfer-
retrainedfromscratchwithinouradversarialframe-
|       |                   |     |     |         |              | enceremainsidenticaltothebaselinemodel. |     |     |     |     |     | Dur- |
| ----- | ----------------- | --- | --- | ------- | ------------ | --------------------------------------- | --- | --- | --- | --- | --- | ---- |
| work. | The discriminator |     | is  | trained | concurrently |                                         |     |     |     |     |     |      |
ingtraining,theprimarycomputationaloverhead
| with | the encoder. | The | discriminator’s |     | role is to |                          |     |     |     |                  |     |     |
| ---- | ------------ | --- | --------------- | --- | ---------- | ------------------------ | --- | --- | --- | ---------------- | --- | --- |
|      |              |     |                 |     |            | arisesfromtwocomponents: |     |     |     | (1)theforwardand |     |     |
effectivelydistinguishbetweenrealandsynthetic
backwardpassesthroughthediscriminator,and(2)
embeddings,enforcingawell-structuredrepresen-
thecomputationoftheAM-Softmaxlossforboth
tation.
|     |     |     |     |     |     | realandsynthetic(mixup)data. |     |     |     | Sincesyntheticem- |     |     |
| --- | --- | --- | --- | --- | --- | ---------------------------- | --- | --- | --- | ----------------- | --- | --- |
Mixupdiscriminator Todeterminethemostin- beddingsaredynamicallygeneratedfromreal-data
formativeHuBERThiddenlayersforspeakerrep- embeddings, no additional embedding computa-
resentation, we conduct an ablation study. We tions are required. In our experiments, synthetic
experiment with different layer configurations, data is generated in a 1:1 ratio with real data, ef-

ID Model #Parameters EER(%) minDCF Utterances #Speakers
1 MFA-Conformer 19.8M 12.82 0.770 12,335 100
2 MFA-Adversarial 19.8M 11.83 0.790 12,335 100
3 MFA-Conformer 19.8M 0.86 0.066 1,240,651 7,205
4 MFA-Adversarial 19.8M 0.81 0.036 1,240,651 7,205
Table5: PerformanceoverviewofallsystemsonVoxCeleb1-O
DATASET BASELINE AT tionwithdifferentmodelconfigurations.
EMOTIONCLASSIFICATION 83% 85.50%
UsingtheMFAConformerastheEncoder,we
HOWFARSPK 77.91% 79.97%
conductseveralexperiments(Table2)reportingthe
Table6: ClassificationaccuraciesforHubertEncoder EER.Theadditionofsyntheticlossalone(Model
baselineandwithAdversarialTrainingontwodifferent ID ) yield slight improvements over the base-
2
tasks.
line (Model ID ). More substantial gains were
1
achievedthroughadversarialtraining(ModelID ),
3
withfurtherimprovementsobservedwhenincorpo-
fectivelyquadruplingthecomputetimefortheloss
ratingthemixupdiscriminator(ModelID ). The
4
calculations(doublingforAM-Softmaxonrealand
best performance was achieved by Model ID ,
6
syntheticdata,anddoublingforthediscriminator).
which combined all three components: synthetic
However,asthelosscomputationconstitutesami-
loss,mixupdiscriminator,andadversarialtraining.
nor fraction of the overall training cost, the total
Tofurthervalidatethegeneralizabilityofourap-
increaseintrainingtimeremainsmodest. Addition-
proach,weimplementitwithanalternativespeaker
ally, since mixup data is computed dynamically,
encoder. AsshowninTable3,thecombinationof
thememoryoverheadisnegligible.
adversarial training and mixup discriminator im-
proved performance by 6.56% compared to the
5 Results&Analysis
baseline. Addingsyntheticlossfurtherenhanced
Tovalidatetheeffectivenessofourapproach,we theimprovementto8.29%.
conductcomprehensiveexperimentsacrossspeaker We conduct an ablation study to identify the
verification,emotionclassification,andspeakerdis- mostinformativehiddenlayersforspeakerrepre-
tanceestimationtasks. Ouranalysisdemonstrates sentation. Table4presentstheEERandminDCF
significant improvements through adversarial re- acrossvariouslayerconfigurations. Ouranalysis
finementonmodelgeneralization. revealed that layers 7, 9, 11, and 12 provide the
most effective speaker characteristics representa-
5.1 SpeakerVerificationTask
tion,suggestingthatlaterlayerscapturemorevalu-
Weevaluate our models onVoxCeleb1-O, theof- ablespeaker-specificinformation.
ficial test set of VoxCeleb1. For evaluating the
5.1.2 LargeScale
performance,weusetheEqualErrorRate(EER)
andminimumDetectionCostFunction(minDCF). To demonstrate scalability, we train on the com-
EER represents the point where the false accep- bined VoxCeleb1 and VoxCeleb2 datasets, creat-
tancerateequalsthefalserejectionrate,providing ingasubstantiallylargertrainingcorpus. Assum-
a single measure of verification accuracy (lower marizedinTable5,theMFA-Conformerencoder
is better). The minDCF quantifies the cost of de- (ModelID 3 )achievesstrongbaselineperformance,
tection errors, balancing false positives and false whichisfurtherenhancedthroughadversarialre-
negatives,withlowervaluesindicatingbetterper- finement(ModelID 4 ).
formance. Thesemetricsassessthemodel’sability Specifically, adversarial training reduces the
todistinguishbetweensame-speakeranddifferent- EERfrom0.86%to0.81%andnearlyhalvesthe
speakerpairseffectively. minDCFfrom0.066to0.036. Theseimprovements
areconsistentwiththesmall-scaletrends,confirm-
5.1.1 SmallScale ingthatourframeworkscalesrobustlytolargerand
Ourinitialevaluationsfocusedonmodelstrained morediversespeakerpopulations.
onVoxCeleb1tofacilitatethoroughexperimenta- Importantly, we note that even in the limited-

diversity setting of VoxCeleb1—where only 100 shotlearning. Futureworkwillfocusonexpanding
speakersarepresent—ourapproachstillyieldscon- CAARMA’sutilitytolargerdatasetsandotherdo-
sistentgains(ModelIDs1–2). Thissuggeststhat mains,suchascomputervision.
| the framework  |     | is not         | merely | leveraging | broader |         |             |     |     |     |     |     |     |
| -------------- | --- | -------------- | ------ | ---------- | ------- | ------- | ----------- | --- | --- | --- | --- | --- | --- |
| class coverage |     | in large-scale |        | datasets,  | but     | is also | Limitations |     |     |     |     |     |     |
effectiveinscenarioswithconstrainedspeakerdi-
TheCAARMAframework,whileshowcasingno-
versity. Inpractice,thismeansthatourmethodim-
|     |     |     |     |     |     |     | table enhancements |     | in  | speaker | verification |     | and |
| --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | --- | ------- | ------------ | --- | --- |
provesspeakerverificationrobustnessbothunder zero-shotlearningtasks,issubjecttoseverallim-
resource-richconditionsandundermorerestrictive
|     |     |     |     |     |     |     | itations | that merit | further | exploration. |     | Although |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | ------- | ------------ | --- | -------- | --- |
dataavailability.
|     |     |     |     |     |     |     | it performs | well | in controlled |     | settings, | its | scala- |
| --- | --- | --- | --- | --- | --- | --- | ----------- | ---- | ------------- | --- | --------- | --- | ------ |
Overall,theseresultshighlightthatadversarial
|     |     |     |     |     |     |     | bility to | extremely | large | or  | diverse | datasets, | as  |
| --- | --- | --- | --- | --- | --- | --- | --------- | --------- | ----- | --- | ------- | --------- | --- |
trainingisbeneficialacrosstrainingregimes: itgen- wellasitsapplicabilitytoreal-worldscenarioswith
eralizeswelltolarge-scale,diversedatasetswhile
|     |     |     |     |     |     |     | highspeakervariability, |     |     | hasyettobefullyestab- |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------------- | --- | --- | --------------------- | --- | --- | --- |
stillofferingtangibleimprovementswhendiversity
|            |     |     |     |     |     |     | lished. Thisalsoaddscomplexitytotheimplemen- |     |     |     |     |     |     |
| ---------- | --- | --- | --- | --- | --- | --- | -------------------------------------------- | --- | --- | --- | --- | --- | --- |
| islimited. |     |     |     |     |     |     | tationandincreasescomputationaldemands,which |     |     |     |     |     |     |
mayrestrictaccessibilityforthosewithlimitedre-
5.2 EmotionandSpeakerDistanceTasks
sources.
Todemonstratethegeneralizabilityofourapproach
EthicsStatement
| across different |                   | speech | processing |         | domains,    | we  |            |     |           |     |              |     |        |
| ---------------- | ----------------- | ------ | ---------- | ------- | ----------- | --- | ---------- | --- | --------- | --- | ------------ | --- | ------ |
| evaluate         | its effectiveness |        | on         | emotion | classifica- |     |            |     |           |     |              |     |        |
|                  |                   |        |            |         |             |     | The CAARMA |     | framework |     | is developed |     | with a |
tionandspeakerdistanceestimationusingtheDai- commitment to ethical considerations, especially
| lyTalkandHowFarAreYoutestsets, |     |     |     |     | respectively. |     |            |         |     |     |           |     |          |
| ------------------------------ | --- | --- | --- | --- | ------------- | --- | ---------- | ------- | --- | --- | --------- | --- | -------- |
|                                |     |     |     |     |               |     | concerning | privacy | and | the | potential | for | surveil- |
AsshowninTable6,ourmethodimprovedclassi-
|                                  |     |     |     |                |     |     | lancemisuse. | Itiscrucialtoensurethatthistech- |     |     |     |     |     |
| -------------------------------- | --- | --- | --- | -------------- | --- | --- | ------------ | -------------------------------- | --- | --- | --- | --- | --- |
| ficationaccuracyacrossbothtasks: |     |     |     | emotionclassi- |     |     |              |                                  |     |     |     |     |     |
nology,whileadvancingthecapabilitiesofspeaker
ficationaccuracyincreasedfrom83%to85.50%, verification systems, is employed within the con-
whilespeakerdistanceestimationimprovedfrom
finesofstrictethicalguidelinesandprivacyregula-
| 77.91%to79.97%. |     | Theseresultsdemonstratethat |     |     |     |     |     |     |     |     |     |     |     |
| --------------- | --- | --------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
tionstopreventanyinvasionofindividualprivacy.
ourapproachcanbeeffectivelyintegratedwithvar- Asthisframeworkfacilitatesthegenerationofsyn-
iousmodelsacrossvariousdomains.
theticdata,wealsofocusonpreventingbiasesthat
couldariseinsyntheticdatasets,ensuringfairrep-
6 Conclusion
|     |     |     |     |     |     |     | resentationacrossdifferentgroups. |     |     |     | Inadherenceto |     |     |
| --- | --- | --- | --- | --- | --- | --- | --------------------------------- | --- | --- | --- | ------------- | --- | --- |
theACLEthicsPolicy,weemphasizetransparency
| In this work, | we  | introduce | CAARMA, |     | a   | novel |     |     |     |     |     |     |     |
| ------------- | --- | --------- | ------- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
inthedeploymentofCAARMAandadvocatefor
classaugmentationframeworkdesignedtotackle
itsuseinethicallyjustifiablemannersthatrespect
| the challenge |     | of limited | class | diversity | in  | zero- |     |     |     |     |     |     |     |
| ------------- | --- | ---------- | ----- | --------- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
individualrightsanddataintegrity.
| shot inference |             | tasks. | Our approach |             | synthesizes |         |     |     |     |     |     |     |     |
| -------------- | ----------- | ------ | ------------ | ----------- | ----------- | ------- | --- | --- | --- | --- | --- | --- | --- |
| strategic      | data mixing |        | with an      | adversarial |             | refine- |     |     |     |     |     |     |     |
mentmechanismtoalignrealandsyntheticclasses
| effectively | within | the embedding |     | space. | We  | vali- |     |     |     |     |     |     |     |
| ----------- | ------ | ------------- | --- | ------ | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
dateCAARMA’seffectivenessinspeakerverifica-
tion,achievingan8%improvementoverbaseline
models,andextenditsapplicationtoemotionclas-
sificationandspeakerdistanceestimation,where
| it also shows | significant |            | gains.     | These     | results   | un-   |     |     |     |     |     |     |     |
| ------------- | ----------- | ---------- | ---------- | --------- | --------- | ----- | --- | --- | --- | --- | --- | --- | --- |
| derscore      | CAARMA’s    |            | capability | to        | enhance   | em-   |     |     |     |     |     |     |     |
| bedding       | structures  | in various |            | zero-shot | inference |       |     |     |     |     |     |     |     |
| scenarios.    | Our         | framework  | offers     | a         | scalable  | solu- |     |     |     |     |     |     |     |
tiontotheclassdiversityproblem,facilitatinginte-
grationintoexistingsystemswithouttheneedfor
| newreal-worlddatacollection. |     |     |     | Withourcodere- |     |     |     |     |     |     |     |     |     |
| ---------------------------- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
leasedpublicly,weanticipatethatCAARMAwill
aidbothresearchandpracticalapplicationsinzero-

References
Wei-NingHsu,BenjaminBolte,Yao-HungHubertTsai,
KushalLakhotia,RuslanSalakhutdinov,andAbdel-
JinwooBaek,HwanjunKim,SeunghyeonJeong,Won-
|     |     |     |     |     |     |     | rahman | Mohamed. | 2021. | Hubert: | Self-supervised |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------ | -------- | ----- | ------- | --------------- | --- | --- |
junChoi,andChulheeKim.2021. Gridmix: Asim- speechrepresentationlearningbymaskedprediction
plegrid-baseddataaugmentationmethodforobject
|            |                                   |     |     |     |     |     | ofhiddenunits. |     | IEEE/ACMtransactionsonaudio, |     |     |     |     |
| ---------- | --------------------------------- | --- | --- | --- | --- | --- | -------------- | --- | ---------------------------- | --- | --- | --- | --- |
| detection. | InProceedingsoftheIEEE/CVFConfer- |     |     |     |     |     |                |     |                              |     |     |     |     |
speech,andlanguageprocessing,29:3451–3460.
enceonComputerVisionandPatternRecognition,
pages3255–3264.
|     |     |     |     |     |     |     | Chien-yu | Huang,         | Ke-Han | Lu, Shih-Heng |     | Wang,    | Chi- |
| --- | --- | --- | --- | --- | --- | --- | -------- | -------------- | ------ | ------------- | --- | -------- | ---- |
|     |     |     |     |     |     |     | Yuan     | Hsiao, Chun-Yi |        | Kuan, Haibin  | Wu, | Siddhant |      |
TingChen,XiaoweiWang,YunzhiShen,WeihaoNie,
Arora,Kai-WeiChang,JiatongShi,YifanPeng,etal.
HanZhang,BinglieLi,andYuZhang.2022. Trans- 2024. Dynamic-superb: Towardsadynamic,collabo-
| mix: | Attending | to  | mix for | vision | transformers. | In  |     |     |     |     |     |     |     |
| ---- | --------- | --- | ------- | ------ | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
rative,andcomprehensiveinstruction-tuningbench-
ProceedingsoftheIEEE/CVFConferenceonCom-
|     |     |     |     |     |     |     | mark | for speech. | In  | ICASSP, | pages 12136–12140. |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---- | ----------- | --- | ------- | ------------------ | --- | --- |
puterVisionandPatternRecognition,pages7367–
IEEE.
7377.
Seong-HyeonHwang,MinsuKim,andStevenEuijong
| J Chung, | A Nagrani,              |     | and | A Zisserman. | 2018.            | Vox- |                                        |     |           |                         |     |        |     |
| -------- | ----------------------- | --- | --- | ------------ | ---------------- | ---- | -------------------------------------- | --- | --------- | ----------------------- | --- | ------ | --- |
|          |                         |     |     |              |                  |      | Whang.2024.                            |     | Rc-mixup: | Adataaugmentationstrat- |     |        |     |
| celeb2:  | Deepspeakerrecognition. |     |     |              | Interspeech2018. |      |                                        |     |           |                         |     |        |     |
|          |                         |     |     |              |                  |      | egyagainstnoisydataforregressiontasks. |     |           |                         |     | InPro- |     |
ceedingsofthe30thACMSIGKDDConferenceon
SamueleCornell,JordanDarefsky,ZhiyaoDuan,and
KnowledgeDiscoveryandDataMining,pages1155–
| ShinjiWatanabe.2024. |     |     | Generatingdatawithtext-to- |     |     |     |     |     |     |     |     |     |     |
| -------------------- | --- | --- | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
1165.
speechandlarge-languagemodelsforconversational
| speechrecognition. |     |     | InProc.SynData4GenAI2024, |     |     |     |     |     |     |     |     |     |     |
| ------------------ | --- | --- | ------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
AmitJindal,NarayananElavathurRanganatha,Aniket
pages6–10.
|     |     |     |     |     |     |     | Didolkar, | Arijit | Ghosh | Chowdhury, | Di  | Jin, Ramit |     |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------ | ----- | ---------- | --- | ---------- | --- |
Ekin D Cubuk, Barret Zoph, Dandelion Mane, Vijay Sawhney, andRajivRatnShah.2020. Speechmix-
|                                         |     |          |     |           |              |        | augmenting           | deep | sound | recognition             |     | using hidden |     |
| --------------------------------------- | --- | -------- | --- | --------- | ------------ | ------ | -------------------- | ---- | ----- | ----------------------- | --- | ------------ | --- |
| Vasudevan,                              |     | and Quoc | V   | Le. 2019. | Autoaugment: |        |                      |      |       |                         |     |              |     |
|                                         |     |          |     |           |              |        | spaceinterpolations. |      |       | InINTERSPEECH,pages861– |     |              |     |
| Learningaugmentationstrategiesfromdata. |     |          |     |           |              | InPro- |                      |      |       |                         |     |              |     |
865.
ceedingsoftheIEEE/CVFconferenceoncomputer
visionandpatternrecognition,pages113–123.
|     |     |     |     |     |     |     | Jang-Hyun | Kim, | Wonho | Choo, | and Hyun | Oh  | Song. |
| --- | --- | --- | --- | --- | --- | --- | --------- | ---- | ----- | ----- | -------- | --- | ----- |
BrechtDesplanques,JentheThienpondt,andKrisDe- 2020. Puzzle mix: Exploiting saliency and local
muynck. 2020. Ecapa-tdnn: Emphasized channel statisticsforoptimalmixup. InInternationalConfer-
enceonMachineLearning(ICML).
attention,propagationandaggregationintdnnbased
| speakerverification. |     |     | Interspeech. |     |     |     |                                         |     |     |     |     |     |     |
| -------------------- | --- | --- | ------------ | --- | --- | --- | --------------------------------------- | --- | --- | --- | --- | --- | --- |
|                      |     |     |              |     |     |     | KaungMyatKyawandJonathanHoyinChan.2024. |     |     |     |     |     | A   |
frameworkforsyntheticaudioconversationsgenera-
| Alexey | Dosovitskiy, |     | Lucas | Beyer, | Alexander |     |     |     |     |     |     |     |     |
| ------ | ------------ | --- | ----- | ------ | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, tionusinglargelanguagemodels. In2024IEEE/WIC
Thomas Unterthiner, Mostafa Dehghani, Matthias InternationalConferenceonWebIntelligenceandIn-
Minderer,GeorgHeigold,SylvainGelly,etal.2021. telligentAgentTechnology(WI-IAT),pages355–359.
IEEE.
| An image  |             | is worth | 16x16 | words: | Transformers     |     |             |          |     |         |         |          |     |
| --------- | ----------- | -------- | ----- | ------ | ---------------- | --- | ----------- | -------- | --- | ------- | ------- | -------- | --- |
| for image | recognition |          | at    | scale. | In International |     |             |          |     |         |         |          |     |
|           |             |          |       |        |                  |     | Jin-Ha Lee, | Muhammad |     | Zaigham | Zaheer, | Marcella |     |
ConferenceonLearningRepresentations.
|     |     |     |     |     |     |     | Astrid, | and Seung-Ik |     | Lee. 2020. | Smoothmix: |     | a   |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------------ | --- | ---------- | ---------- | --- | --- |
Sreyan Ghosh, Sonal Kumar, Zhifeng Kong, Rafael simple yet effective data augmentation to train ro-
Valle,BryanCatanzaro,andDineshManocha.2026. bustclassifiers. In2020IEEE/CVFConferenceon
Synthio: Augmenting small-scale audio classifica- ComputerVisionandPatternRecognitionWorkshops
(CVPRW),pages3264–3274.IEEEComputerSoci-
| tiondatasetswithsyntheticdata. |     |     |     |     | InTheThirteenth |     |     |     |     |     |     |     |     |
| ------------------------------ | --- | --- | --- | --- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- |
ety.
| International |     | Conference |     | on Learning | Representa- |     |     |     |     |     |     |     |     |
| ------------- | --- | ---------- | --- | ----------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
tions.
QiujiaLi,FlorianLKreyssig,ChaoZhang,andPhilipC
NileshGupta,SakinaBohra,YashotejaPrabhu,Saurabh Woodland.2021. Discriminativeneuralclusteringfor
Purohit,andManikVarma.2021. Generalizedzero- speakerdiarisation. In2021IEEEspokenlanguage
shotextrememulti-labellearning. InProceedingsof technologyworkshop(SLT),pages574–581.IEEE.
the27thACMSIGKDDConferenceonKnowledge
JihaoLiu,BoxiaoLiu,HangZhou,HongshengLi,and
Discovery&DataMining,pages527–535.
|     |     |     |     |     |     |     | YuLiu.2022. |     | Tokenmix: | Rethinkingimagemixing |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | --------- | --------------------- | --- | --- | --- |
ZongyanHan,ZhenyongFu,ShuoChen,andJianYang. for data augmentation in vision transformers. In
2021. Contrastiveembeddingforgeneralizedzero- Europeanconferenceoncomputervision,pages455–
| shotlearning. |     | InProceedingsoftheIEEE/CVFcon- |     |     |     |     | 471.Springer. |     |     |     |     |     |     |
| ------------- | --- | ------------------------------ | --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- |
ferenceoncomputervisionandpatternrecognition,
ShaoboMin,HantaoYao,HongtaoXie,Zheng-JunZha,
pages2371–2381.
|     |     |     |     |     |     |     | and Yongdong |     | Zhang. | 2019. Domain-specific |     |     | em- |
| --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ------ | --------------------- | --- | --- | --- |
Dan Hendrycks and Kevin Gimpel. 2016. Gaus- beddingnetworkforzero-shotrecognition. InPro-
sian error linear units (gelus). arXiv preprint ceedingsofthe27thACMInternationalConference
| arXiv:1606.08415. |     |     |     |     |     |     | onMultimedia,pages2070–2078. |     |     |     |     |     |     |
| ----------------- | --- | --- | --- | --- | --- | --- | ---------------------------- | --- | --- | --- | --- | --- | --- |

Shaobo Min, Hantao Yao, Hongtao Xie, Zheng-Jun HuaxiuYao,YipingWang,LinjunZhang,JamesYZou,
Zha,andYongdongZhang.2020. Domain-oriented andChelseaFinn.2022. C-mixup: Improvinggener-
semantic embedding for zero-shot learning. IEEE alizationinregression. Advancesinneuralinforma-
TransactionsonMultimedia,23:3919–3930. tionprocessingsystems,35:3361–3376.
|               |     |               |     |                  |     | Sangdoo Yun, | Dongyoon Han, | Seong Joon | Oh, |
| ------------- | --- | ------------- | --- | ---------------- | --- | ------------ | ------------- | ---------- | --- |
| ArshaNagrani, |     | JoonSonChung, |     | andAndrewZisser- |     |              |               |            |     |
man.2017. Voxceleb: Alarge-scalespeakeridentifi- SanghyukChun,JunsukChoe,andYoungjoonYoo.
cationdataset. Interspeech2017. 2019. Cutmix:Regularizationstrategytotrainstrong
|     |     |     |     |     |     | classifierswithlocalizablefeatures. |     | InProceedings |     |
| --- | --- | --- | --- | --- | --- | ----------------------------------- | --- | ------------- | --- |
DanielSPark,WilliamChan,YuZhang,Chung-Cheng oftheIEEE/CVFinternationalconferenceoncom-
Chiu,BarretZoph,EkinDCubuk,andQuocVLe. putervision,pages6023–6032.
| 2019. | Specaugment: |     | A simple | data augmentation |     |     |     |     |     |
| ----- | ------------ | --- | -------- | ----------------- | --- | --- | --- | --- | --- |
HongyiZhang,MoustaphaCisse,YannNDauphin,and
| methodforautomaticspeechrecognition. |     |     |     |     | InInter- |                      |        |                 |     |
| ------------------------------------ | --- | --- | --- | --- | -------- | -------------------- | ------ | --------------- | --- |
|                                      |     |     |     |     |          | DavidLopez-Paz.2018. | Mixup: | Beyondempirical |     |
speech,pages2613–2617.
|     |     |     |     |     |     | riskminimization. | arXivpreprintarXiv:1710.09412. |     |     |
| --- | --- | --- | --- | --- | --- | ----------------- | ------------------------------ | --- | --- |
FarhadPourpanah,MoloudAbdar,YuxuanLuo,Xinlei
Zhou, RanWang, CheePengLim, Xi-ZhaoWang, YangZhang,ZhiqiangLv,HaibinWu,ShanshanZhang,
and QM Jonathan Wu. 2022. A review of gener- PengfeiHu,ZhiyongWu,Hung-yiLee,andHelen
|        |           |          |          |      |          | Meng. 2022. | Mfa-conformer: | Multi-scale | feature |
| ------ | --------- | -------- | -------- | ---- | -------- | ----------- | -------------- | ----------- | ------- |
| alized | zero-shot | learning | methods. | IEEE | transac- |             |                |             |         |
aggregationconformerforautomaticspeakerverifi-
tionsonpatternanalysisandmachineintelligence,
|     |     |     |     |     |     | cation. Interspeech. |     |     |     |
| --- | --- | --- | --- | --- | --- | -------------------- | --- | --- | --- |
45(4):4051–4070.
PengkaiZhu,HanxiaoWang,andVenkateshSaligrama.
JieQin,JieminFang,QianZhang,WenyuLiu,Xingang
Wang,andXinggangWang.2020. Resizemix: Mix- 2019. Generalized zero-shot recognition based on
|     |     |     |     |     |     | visuallysemanticembedding. |     | InProceedingsofthe |     |
| --- | --- | --- | --- | --- | --- | -------------------------- | --- | ------------------ | --- |
ingdatawithpreservedobjectinformationandtrue
IEEE/CVFConferenceonComputerVisionandPat-
| labels. | arXive-prints,pagesarXiv–2012. |     |     |     |     |     |     |     |     |
| ------- | ------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
ternRecognition(CVPR).
LakshmiVenkataramanan,AditiRaghunathan,Ananya
| Kapoor,andGunjanJoshi.2022. |     |     |     | Alignmix: | Improv- |     |     |     |     |
| --------------------------- | --- | --- | --- | --------- | ------- | --- | --- | --- | --- |
ingconsistencyandrobustnessinvisiontransformers
| viaalignedmixupregularization. |     |     |     | InProceedingsof |     |     |     |     |     |
| ------------------------------ | --- | --- | --- | --------------- | --- | --- | --- | --- | --- |
theIEEE/CVFConferenceonComputerVisionand
PatternRecognition,pages10338–10347.
VikasVerma,AlexLamb,ChristopherBeckham,Amir
| Najafi,            | Ioannis | Mitliagkas,      | David          | Lopez-Paz, | and        |     |     |     |     |
| ------------------ | ------- | ---------------- | -------------- | ---------- | ---------- | --- | --- | --- | --- |
| YoshuaBengio.2019. |         |                  | Manifoldmixup: |            | Betterrep- |     |     |     |     |
| resentations       |         | by interpolating | hidden         | states.    | In In-     |     |     |     |     |
ternationalconferenceonmachinelearning,pages
6438–6447.PMLR.
| Li Wan, | Quan          | Wang, Alan  | Papir,  | and Ignacio        | Lopez    |     |     |     |     |
| ------- | ------------- | ----------- | ------- | ------------------ | -------- | --- | --- | --- | --- |
| Moreno. | 2018.         | Generalized |         | end-to-end         | loss for |     |     |     |     |
| speaker | verification. |             | In 2018 | IEEE International |          |     |     |     |     |
ConferenceonAcoustics,SpeechandSignalProcess-
ing(ICASSP),pages4879–4883.IEEE.
FengWang,JianCheng,WeiyangLiu,andHaijunLiu.
2018. Additivemarginsoftmaxforfaceverification.
IEEESignalProcessingLetters,25(7):926–930.
| Y Xian, | CH Lampert, | B   | Schiele, | and Z | Akata. 2018. |     |     |     |     |
| ------- | ----------- | --- | -------- | ----- | ------------ | --- | --- | --- | --- |
Zero-shotlearning—acomprehensiveevaluationof
| the good, | the | bad and | the ugly. | arXiv | preprint |     |     |     |     |
| --------- | --- | ------- | --------- | ----- | -------- | --- | --- | --- | --- |
arXiv:1707.00600.
Guo-SenXie,ZhengZhang,HuanXiong,LingShao,
| andXuelongLi.2022. |                                      |        | Towardszero-shotlearning: |     |           |     |     |     |     |
| ------------------ | ------------------------------------ | ------ | ------------------------- | --- | --------- | --- | --- | --- | --- |
| A brief            | review                               | and an | attention-based           |     | embedding |     |     |     |     |
| network.           | IEEETransactionsonCircuitsandSystems |        |                           |     |           |     |     |     |     |
forVideoTechnology,33(3):1181–1197.
| Bing Xu.  | 2015.            | Empirical | evaluation | of    | rectified ac- |     |     |     |     |
| --------- | ---------------- | --------- | ---------- | ----- | ------------- | --- | --- | --- | --- |
| tivations | in convolutional |           | network.   | arXiv | preprint      |     |     |     |     |
arXiv:1505.00853.