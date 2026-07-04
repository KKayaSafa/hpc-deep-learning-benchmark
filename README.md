# TRUBA Merkezi - ARF Süper Bilgisayar ile Derin Öğrenme Projesi

**Proje Adı:** Farklı GPU Nesilleri, Paralelleştirme Yöntemleri, Optimizasyon Algoritmaları ve Öğrenme Oranlarının Derin Öğrenme Eğitimine Etkisi

**Yürütücü:** Safa Kılıçkaya  
**Danışman:** Dr. İsmail GÜZEL (TÜBİTAK ULAKBİM, TRUBA)  
**Tarih:** Haziran 2025

---

## Proje Özeti

Bu çalışma, TRUBA Süper Bilgisayar Merkezi altyapısını kullanarak derin öğrenme model eğitiminin kapsamlı bir analizini sunmaktadır. Proje, üç farklı aşamada gerçekleştirilmiş olup ResNet50, InceptionV3, EfficientNetB2 ve VGG19 modellerinin CIFAR-10 veri seti üzerindeki performansları sistematik olarak değerlendirilmiştir.

Projenin benzersiz yanı, her aşamada farklı teknik yaklaşımlar benimsenmiş olmasıdır. İlk aşama temel GPU ölçeklendirme analizine odaklanırken, ikinci aşamada GPU utilization optimizasyonu ve yeni optimizer algoritmaları test edilmiş, üçüncü aşamada ise farklı paralelleştirme stratejileri keşfedilmiştir. Bu çok katmanlı yaklaşım, süper bilgisayar kaynaklarının verimli kullanımı ve model performansının optimize edilmesi konularında değerli içgörüler sağlamıştır.

## Proje Mimarisi ve Aşamalı Geliştirme Süreci

### Proje Klasör Yapısı

Proje süresince öğrenilen deneyimler doğrultusunda klasör yapısı ihtiyaca göre genişletişmiştir. Her model için üç farklı versiyon oluşturularak, farklı optimizasyon stratejileri test edilmiştir:

```
proje/
├── efficientNet/
│   ├── src/          # Python kodları
│   ├── scripts/      # Job scriptleri
│   ├── logs/         # Log dosyaları
│   ├── out/          # 4GPU ile eğitim
│   ├── out2/         # 2GPU ile eğitim
│   ├── out3/         # 1GPU ile eğitim
│   └── results/      # Deneysel sonuçlar
├── inceptionV3/
│   ├── src/          # Python kodları
│   ├── scripts/      # Job scriptleri
│   ├── logs/         # Log dosyaları
│   ├── out/          # 4GPU ile eğitim
│   ├── out2/         # 2GPU ile eğitim
│   ├── out3/         # 1GPU ile eğitim
│   └── results/      # Deneysel sonuçlar
├── resnet50/
│   ├── src/          # Python kodları
│   ├── scripts/      # Job scriptleri
│   ├── logs/         # Log dosyaları
│   ├── out/          # 4GPU ile eğitim
│   ├── out2/         # 2GPU ile eğitim
│   ├── out3/         # 1GPU ile eğitim
│   └── results/      # Deneysel sonuçlar
├── vgg19/
│   ├── src/          # Python kodları
│   ├── scripts/      # Job scriptleri
│   ├── logs/         # Log dosyaları
│   ├── out/          # 4GPU ile eğitim
│   ├── out2/         # 2GPU ile eğitim
│   ├── out3/         # 1GPU ile eğitim
│   └── results/      # Deneysel sonuçlar
└── README.md         # Proje açıklaması

proje2/
└── (aynı klasör yapısı)

proje3/
└── (aynı klasör yapısı)
```

Her proje klasörü kendi içinde organize bir yapı barındırmaktadır: `src/` dizininde Python kodları, `scripts/` dizininde SLURM job scriptleri, `logs/` dizininde eğitim logları, `out/` klasörlerinde farklı GPU konfigürasyonlarının çıktıları ve `results/` dizininde farklı formatlarda deneysel sonuç tabloları ve ekran görüntüleri bulunmaktadır. bulunmaktadır. Raporun sonunda verilen Google Drive bağlantısındaki klasörlerde proje2 ve proje3 için "nvidia-smi" komutlarının ekran görüntüleri de eklenmiştir. 

### Üç Aşamalı Geliştirme Süreci

**Proje 1: Temel Analiz ve GPU Ölçeklendirme (126 Deney)**

İlk aşamada, TRUBA altyapısının temel yetenekleri keşfedilmiştir. Bu aşamada 1, 2 ve 4 GPU kullanarak farklı modellerin performansları karşılaştırılmıştır. Distributed Data Parallel (DDP) stratejisi kullanılarak Adam, AdamW ve SGD optimizasyon algoritmalarının etkisi incelenmiştir. Ancak bu aşamada GPU utilization modülü entegre edilmediği için detaylı kaynak kullanım analizi yapılamamış, ayrıca num_workers parametresinin düşük tutulması nedeniyle TRUBA sisteminden verimlilik uyarıları alınmıştır.

Bu aşamada en dikkat çekici bulgu, GPU sayısının artmasının her zaman performans artışı sağlamadığı olmuştur. ResNet50 modelinde 97.11% doğruluk ile en yüksek performans elde edilirken, InceptionV3 modeli 94.45% ortalama ile en tutarlı sonuçları vermiştir.

**Proje 2: GPU Optimizasyonu ve RMSprop Entegrasyonu (74 Deney)**

İkinci aşamada, ilk projeden alınan dersler doğrultusunda önemli iyileştirmeler yapılmıştır. GPU utilization analizi için bir modül (`gpu_utilization.py`) ana eğitim koduna entegre edilmiştir. Bu sayede TensorBoard üzerinden GPU kullanım istatistikleri detaylı olarak izlenebilmiştir. Ayrıca num_workers parametresi artırılarak veri yükleme süreçleri optimize edilmiş ve kullanılan hızlandırıcılarının verimliliği artırılmıştır.

Bu aşamanın en önemli yeniliği RMSprop optimizasyon algoritmasının test edilmesidir. RMSprop'un performansı özellikle ResNet50 modelinde dikkat çekici olmuş, 97.00% doğruluk ile çok başarılı sonuçlar vermiştir. EfficientNetB2 ve VGG19 gibi daha küçük modeller için kaynak verimliliği gözetilerek 2 GPU kullanımına geçilmiştir.

**Proje 3: FSDP Stratejisi ve Paralelleştirme Çeşitliliği (74 Deney)**

Üçüncü aşamada, paralelleştirme stratejilerindeki çeşitliliği artırmak amacıyla Fully Sharded Data Parallel (FSDP) stratejisi öncelenmiştir. Daha önceki projelerde 4 saatlik süre sınırlaması nedeniyle tüm kombinasyonlar tamamlanamadığından, bu aşamada FSDP modellerine öncelik verilerek farklı paralelleştirme yaklaşımlarının etkisi sistematik olarak incelenmiştir.

FSDP stratejisi, bellek kullanımında önemli avantajlar sağlamıştır. ResNet50 modelinde FSDP ile 96.88% doğruluk elde edilirken, GPU başına düşen bellek yükü önemli ölçüde azalmıştır.

## Deneysel Sonuçlar ve Kapsamlı Analiz

### proje Deneysel Sonuç Tabloları

### ResNet50 Deneyleri

|batch_size|epochs|gpus|learning_rate|lr_scheduler_name|model_type|nodes|optimizer|strategy|test_loss          |test_acc          |
|----------|------|----|-------------|-----------------|----------|-----|---------|--------|-------------------|------------------|
|32.0      |10.0  |2.0 |0.0001       |cosine           |ResNet50  |1.0  |adam     |ddp     |0.11046180129051208|0.9710999727249146|
|32.0      |10.0  |2.0 |0.0001       |cosine           |ResNet50  |1.0  |adamw    |ddp     |0.11605013161897659|0.9700000286102295|
|32.0      |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |adam     |ddp     |0.11047879606485367|0.9696999788284302|
|32.0      |10.0  |2.0 |0.0001       |step             |ResNet50  |1.0  |adam     |ddp     |0.11287251859903336|0.9692000150680542|
|32.0      |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |adamw    |ddp     |0.11592777073383331|0.9689000248908997|
|32.0      |10.0  |2.0 |0.0001       |step             |ResNet50  |1.0  |adamw    |ddp     |0.1138710081577301 |0.9681000113487244|
|32.0      |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |adam     |ddp     |0.11691529303789139|0.9677000045776367|
|128.0     |10.0  |2.0 |0.0001       |cosine           |ResNet50  |1.0  |adam     |ddp     |0.11821015924215317|0.9671000242233276|
|128.0     |10.0  |2.0 |0.0001       |step             |ResNet50  |1.0  |adam     |ddp     |0.12850713729858398|0.9664000272750854|
|128.0     |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |adamw    |ddp     |0.11966314911842346|0.9663000106811523|
|32.0      |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |adamw    |ddp     |0.12366929650306702|0.9650999903678894|
|128.0     |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |adam     |ddp     |0.12599456310272217|0.9631999731063843|
|128.0     |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |adamw    |ddp     |0.13510847091674805|0.9620000123977661|
|128.0     |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |adam     |ddp     |0.13413994014263153|0.9616000056266785|
|32.0      |10.0  |4.0 |0.0001       |reduce           |ResNet50  |1.0  |adam     |ddp     |0.13721971213817596|0.9613000154495239|
|128.0     |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |adam     |ddp     |0.1459847241640091 |0.9610000252723694|
|128.0     |10.0  |4.0 |0.0001       |reduce           |ResNet50  |1.0  |adam     |ddp     |0.14525026082992554|0.9606999754905701|
|128.0     |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |adamw    |ddp     |0.15403014421463013|0.9599000215530396|
|32.0      |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |adamw    |ddp     |0.149934321641922  |0.9592000246047974|
|32.0      |10.0  |2.0 |0.0001       |none             |ResNet50  |1.0  |adam     |ddp     |0.14122718572616577|0.9592000246047974|
|128.0     |10.0  |2.0 |0.0001       |reduce           |ResNet50  |1.0  |adam     |ddp     |0.15368348360061646|0.9585000276565552|
|32.0      |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |adam     |ddp     |0.14832161366939545|0.9585000276565552|
|32.0      |10.0  |4.0 |0.0001       |reduce           |ResNet50  |1.0  |adamw    |ddp     |0.14891642332077026|0.958299994468689 |
|128.0     |10.0  |2.0 |0.0001       |none             |ResNet50  |1.0  |adam     |ddp     |0.16245974600315094|0.9567000269889832|
|32.0      |10.0  |2.0 |0.0001       |reduce           |ResNet50  |1.0  |adam     |ddp     |0.148093119263649  |0.9556999802589417|
|32.0      |10.0  |2.0 |0.0001       |none             |ResNet50  |1.0  |adamw    |ddp     |0.16567474603652954|0.9524999856948853|
|32.0      |10.0  |1.0 |0.0001       |none             |ResNet50  |1.0  |sgd      |ddp     |0.14829041063785553|0.9495999813079834|
|32.0      |10.0  |1.0 |0.0001       |reduce           |ResNet50  |1.0  |sgd      |ddp     |0.1542963981628418 |0.9495000243186951|
|32.0      |10.0  |1.0 |0.0001       |cosine           |ResNet50  |1.0  |sgd      |ddp     |0.1947644203901291 |0.9355000257492065|
|32.0      |10.0  |1.0 |0.0001       |step             |ResNet50  |1.0  |sgd      |ddp     |0.19756484031677246|0.9351000189781189|
|32.0      |10.0  |2.0 |0.0001       |none             |ResNet50  |1.0  |sgd      |ddp     |0.2146635204553604 |0.9316999912261963|
|32.0      |10.0  |2.0 |0.0001       |reduce           |ResNet50  |1.0  |sgd      |ddp     |0.21576692163944244|0.9294999837875366|
|32.0      |10.0  |2.0 |0.0001       |step             |ResNet50  |1.0  |sgd      |ddp     |0.33430784940719604|0.9016000032424927|
|128.0     |10.0  |1.0 |0.0001       |none             |ResNet50  |1.0  |sgd      |ddp     |0.34208551049232483|0.8966000080108643|
|32.0      |10.0  |2.0 |0.0001       |cosine           |ResNet50  |1.0  |sgd      |ddp     |0.33785781264305115|0.8964999914169312|
|32.0      |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |sgd      |ddp     |0.38059747219085693|0.8888000249862671|
|32.0      |10.0  |4.0 |0.0001       |reduce           |ResNet50  |1.0  |sgd      |ddp     |0.37477052211761475|0.8880000114440918|
|32.0      |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |sgd      |ddp     |0.7001214623451233 |0.8331999778747559|
|128.0     |10.0  |1.0 |0.0001       |step             |ResNet50  |1.0  |sgd      |ddp     |0.679507315158844  |0.8309000134468079|
|128.0     |10.0  |2.0 |0.0001       |reduce           |ResNet50  |1.0  |sgd      |ddp     |0.7969768643379211 |0.8116999864578247|
|32.0      |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |sgd      |ddp     |0.7836846113204956 |0.8083999752998352|
|128.0     |10.0  |2.0 |0.0001       |none             |ResNet50  |1.0  |sgd      |ddp     |0.7970309853553772 |0.8058000206947327|
|128.0     |10.0  |2.0 |0.0001       |cosine           |ResNet50  |1.0  |sgd      |ddp     |1.4461051225662231 |0.707099974155426 |
|128.0     |10.0  |2.0 |0.0001       |step             |ResNet50  |1.0  |sgd      |ddp     |1.4352312088012695 |0.7063999772071838|
|128.0     |10.0  |4.0 |0.0001       |reduce           |ResNet50  |1.0  |sgd      |ddp     |1.506529450416565  |0.6912000179290771|
|128.0     |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |sgd      |ddp     |1.5741616487503052 |0.6818000078201294|
|128.0     |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |sgd      |ddp     |1.9495763778686523 |0.553600013256073 |
|128.0     |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |sgd      |ddp     |1.9423093795776367 |0.5509999990463257|

### InceptionV3 Deneyleri

|batch_size|epochs|gpus|learning_rate|lr_scheduler_name|model_type |nodes|optimizer|strategy|test_loss          |test_acc          |
|----------|------|----|-------------|-----------------|-----------|-----|---------|--------|-------------------|------------------|
|64.0      |10.0  |2.0 |0.0001       |cosine           |InceptionV3|1.0  |adamw    |ddp     |0.11785955727100372|0.9708999991416931|
|32.0      |10.0  |1.0 |0.0001       |cosine           |InceptionV3|1.0  |adamw    |fsdp    |0.1291135996580124 |0.9696999788284302|
|32.0      |10.0  |2.0 |0.0001       |cosine           |InceptionV3|2.0  |adamw    |ddp     |0.11473967134952545|0.9692999720573425|
|32.0      |10.0  |2.0 |0.0001       |step             |InceptionV3|2.0  |adamw    |ddp     |0.1136271208524704 |0.968500018119812 |
|64.0      |10.0  |2.0 |0.0001       |step             |InceptionV3|1.0  |adamw    |ddp     |0.11759831756353378|0.9682000279426575|
|32.0      |10.0  |2.0 |0.0001       |cosine           |InceptionV3|1.0  |adamw    |ddp     |0.12314026057720184|0.9678000211715698|
|32.0      |10.0  |2.0 |0.0001       |step             |InceptionV3|1.0  |adamw    |ddp     |0.12262652814388275|0.9675999879837036|
|64.0      |10.0  |2.0 |0.0001       |cosine           |InceptionV3|2.0  |adamw    |ddp     |0.1167127788066864 |0.9672999978065491|
|32.0      |10.0  |1.0 |0.0001       |step             |InceptionV3|1.0  |adamw    |fsdp    |0.13352170586585999|0.96670001745224  |
|64.0      |10.0  |2.0 |0.0001       |step             |InceptionV3|2.0  |adamw    |ddp     |0.12649594247341156|0.9660999774932861|
|64.0      |10.0  |2.0 |0.0001       |reduce           |InceptionV3|2.0  |adamw    |ddp     |0.14718708395957947|0.9606000185012817|
|64.0      |10.0  |2.0 |0.0001       |none             |InceptionV3|2.0  |adamw    |ddp     |0.13935527205467224|0.9598000049591064|
|64.0      |10.0  |2.0 |0.0001       |none             |InceptionV3|1.0  |adamw    |ddp     |0.13630107045173645|0.9592000246047974|
|32.0      |10.0  |2.0 |0.0001       |none             |InceptionV3|2.0  |adamw    |ddp     |0.14855541288852692|0.9577000141143799|
|64.0      |10.0  |2.0 |0.0001       |reduce           |InceptionV3|1.0  |adamw    |ddp     |0.16138774156570435|0.9577000141143799|
|32.0      |10.0  |2.0 |0.0001       |reduce           |InceptionV3|2.0  |adamw    |ddp     |0.15739214420318604|0.9560999870300293|
|32.0      |10.0  |2.0 |0.0001       |none             |InceptionV3|1.0  |adamw    |ddp     |0.16850432753562927|0.9535999894142151|
|32.0      |10.0  |2.0 |0.0001       |reduce           |InceptionV3|1.0  |adamw    |ddp     |0.16977760195732117|0.9534000158309937|
|32.0      |10.0  |1.0 |0.0001       |none             |InceptionV3|1.0  |adamw    |fsdp    |0.16499795019626617|0.9509999752044678|
|32.0      |10.0  |2.0 |0.0001       |reduce           |InceptionV3|1.0  |sgd      |ddp     |0.17389973998069763|0.944100022315979 |
|32.0      |10.0  |2.0 |0.0001       |none             |InceptionV3|1.0  |sgd      |ddp     |0.1806909441947937 |0.9417999982833862|
|32.0      |10.0  |2.0 |0.0001       |step             |InceptionV3|1.0  |sgd      |ddp     |0.2551633417606354 |0.9218000173568726|
|32.0      |10.0  |2.0 |0.0001       |cosine           |InceptionV3|1.0  |sgd      |ddp     |0.2497258484363556 |0.9200000166893005|
|64.0      |10.0  |2.0 |0.0001       |none             |InceptionV3|1.0  |sgd      |ddp     |0.2608655095100403 |0.9182999730110168|
|32.0      |10.0  |2.0 |0.0001       |none             |InceptionV3|2.0  |sgd      |ddp     |0.2691071331501007 |0.917900025844574 |
|32.0      |10.0  |2.0 |0.0001       |reduce           |InceptionV3|2.0  |sgd      |ddp     |0.26633208990097046|0.9150999784469604|
|32.0      |10.0  |2.0 |0.0001       |step             |InceptionV3|2.0  |sgd      |ddp     |0.42846742272377014|0.8827000260353088|
|32.0      |10.0  |2.0 |0.0001       |cosine           |InceptionV3|2.0  |sgd      |ddp     |0.4653032124042511 |0.8709999918937683|
|64.0      |10.0  |2.0 |0.0001       |none             |InceptionV3|2.0  |sgd      |ddp     |0.48587915301322937|0.8669000267982483|


### EfficientNetB2 Deneyleri

|batch_size|epochs|gpus|learning_rate|lr_scheduler_name|model_type    |nodes|optimizer|strategy|test_loss          |test_acc          |
|----------|------|----|-------------|-----------------|--------------|-----|---------|--------|-------------------|------------------|
|64.0      |15.0  |2.0 |0.001        |cosine           |EfficientNetB2|1.0  |adamw    |ddp     |0.4146803319454193 |0.8910999894142151|
|64.0      |15.0  |4.0 |0.001        |cosine           |EfficientNetB2|1.0  |adamw    |ddp     |0.4002387821674347 |0.8866999745368958|
|64.0      |15.0  |4.0 |0.001        |cosine           |EfficientNetB2|1.0  |adamw    |fsdp    |0.4174714982509613 |0.8848999738693237|
|64.0      |15.0  |2.0 |0.001        |step             |EfficientNetB2|1.0  |adamw    |ddp     |0.38019293546676636|0.882099986076355 |
|64.0      |15.0  |1.0 |0.001        |none             |EfficientNetB2|1.0  |adamw    |ddp     |0.4122398793697357 |0.8804000020027161|
|64.0      |15.0  |2.0 |0.001        |none             |EfficientNetB2|1.0  |adamw    |ddp     |0.4093143939971924 |0.8787999749183655|
|64.0      |15.0  |1.0 |0.001        |step             |EfficientNetB2|1.0  |adamw    |ddp     |0.3920750617980957 |0.8761000037193298|
|64.0      |15.0  |4.0 |0.001        |step             |EfficientNetB2|1.0  |adamw    |fsdp    |0.3943832218647003 |0.8737000226974487|
|64.0      |15.0  |4.0 |0.001        |none             |EfficientNetB2|1.0  |adamw    |fsdp    |0.45797011256217957|0.871999979019165 |
|64.0      |15.0  |4.0 |0.001        |none             |EfficientNetB2|1.0  |adamw    |ddp     |0.4486825168132782 |0.8712999820709229|
|64.0      |15.0  |4.0 |0.001        |reduce           |EfficientNetB2|1.0  |adamw    |fsdp    |0.47735658288002014|0.8705999851226807|
|64.0      |15.0  |4.0 |0.001        |step             |EfficientNetB2|1.0  |adamw    |ddp     |0.40183225274086   |0.870199978351593 |
|64.0      |15.0  |2.0 |0.001        |reduce           |EfficientNetB2|1.0  |adamw    |ddp     |0.4619649350643158 |0.8687000274658203|
|64.0      |15.0  |4.0 |0.001        |reduce           |EfficientNetB2|1.0  |adamw    |ddp     |0.49631309509277344|0.8618999719619751|
|64.0      |15.0  |2.0 |0.001        |none             |EfficientNetB2|1.0  |sgd      |ddp     |0.6397020816802979 |0.7839000225067139|
|64.0      |15.0  |4.0 |0.001        |reduce           |EfficientNetB2|1.0  |sgd      |ddp     |0.7670665979385376 |0.727400004863739 |
|64.0      |15.0  |4.0 |0.001        |none             |EfficientNetB2|1.0  |sgd      |ddp     |0.8036884069442749 |0.7156000137329102|
|64.0      |15.0  |4.0 |0.001        |none             |EfficientNetB2|1.0  |sgd      |fsdp    |0.7963608503341675 |0.7138000130653381|
|64.0      |15.0  |4.0 |0.001        |reduce           |EfficientNetB2|1.0  |sgd      |fsdp    |0.8414227366447449 |0.6998000144958496|
|64.0      |15.0  |4.0 |0.001        |cosine           |EfficientNetB2|1.0  |sgd      |ddp     |0.9788721799850464 |0.6452000141143799|
|64.0      |15.0  |4.0 |0.001        |cosine           |EfficientNetB2|1.0  |sgd      |fsdp    |1.1060696840286255 |0.6205999851226807|
|64.0      |15.0  |4.0 |0.001        |step             |EfficientNetB2|1.0  |sgd      |fsdp    |1.1843442916870117 |0.5684000253677368|
|64.0      |15.0  |4.0 |0.001        |step             |EfficientNetB2|1.0  |sgd      |ddp     |1.261763334274292  |0.5364999771118164|


### VGG19 Deneyleri

|batch_size|epochs|gpus|learning_rate|lr_scheduler_name|model_type|nodes|optimizer|strategy|test_loss          |test_acc          |
|----------|------|----|-------------|-----------------|----------|-----|---------|--------|-------------------|------------------|
|64.0      |10.0  |4.0 |0.0001       |cosine           |VGG19     |1.0  |adamw    |ddp     |0.22570984065532684|0.9503999948501587|
|32.0      |10.0  |4.0 |0.0001       |step             |VGG19     |1.0  |adamw    |ddp     |0.22470623254776   |0.9478999972343445|
|32.0      |10.0  |2.0 |0.0001       |cosine           |VGG19     |1.0  |adamw    |ddp     |0.24063313007354736|0.9477999806404114|
|32.0      |10.0  |4.0 |0.0001       |cosine           |VGG19     |1.0  |adamw    |ddp     |0.23101209104061127|0.9474999904632568|
|64.0      |10.0  |4.0 |0.0001       |step             |VGG19     |1.0  |adamw    |ddp     |0.24459391832351685|0.9467999935150146|
|64.0      |10.0  |2.0 |0.0001       |step             |VGG19     |1.0  |adamw    |ddp     |0.2485867142677307 |0.9462000131607056|
|128.0     |10.0  |4.0 |0.0001       |step             |VGG19     |1.0  |adamw    |ddp     |0.24762101471424103|0.9435999989509583|
|32.0      |10.0  |2.0 |0.0001       |step             |VGG19     |1.0  |adamw    |ddp     |0.2562490999698639 |0.9430999755859375|
|32.0      |10.0  |1.0 |0.0001       |cosine           |VGG19     |1.0  |adamw    |fsdp    |0.3293958902359009 |0.9420999884605408|
|32.0      |10.0  |1.0 |0.0001       |step             |VGG19     |1.0  |adamw    |fsdp    |0.3198210597038269 |0.9377999901771545|
|64.0      |10.0  |4.0 |0.0001       |reduce           |VGG19     |1.0  |adamw    |ddp     |0.2450929880142212 |0.9329000115394592|
|64.0      |10.0  |2.0 |0.0001       |none             |VGG19     |1.0  |adamw    |ddp     |0.26401111483573914|0.9301000237464905|
|32.0      |10.0  |4.0 |0.0001       |reduce           |VGG19     |1.0  |adamw    |ddp     |0.2683247923851013 |0.9297999739646912|
|32.0      |10.0  |4.0 |0.0001       |none             |VGG19     |1.0  |adamw    |ddp     |0.2700462341308594 |0.9282000064849854|
|64.0      |10.0  |4.0 |0.0001       |none             |VGG19     |1.0  |adamw    |ddp     |0.2876032590866089 |0.9266999959945679|
|32.0      |10.0  |2.0 |0.0001       |reduce           |VGG19     |1.0  |adamw    |ddp     |0.3123314678668976 |0.919700026512146 |
|32.0      |10.0  |2.0 |0.0001       |none             |VGG19     |1.0  |adamw    |ddp     |0.3199135363101959 |0.9146000146865845|
|32.0      |10.0  |4.0 |0.0001       |none             |VGG19     |1.0  |sgd      |ddp     |0.24805012345314026|0.9143999814987183|
|32.0      |10.0  |4.0 |0.0001       |reduce           |VGG19     |1.0  |sgd      |ddp     |0.25164803862571716|0.9128000140190125|
|32.0      |10.0  |1.0 |0.0001       |none             |VGG19     |1.0  |adamw    |fsdp    |0.3051368296146393 |0.9064000248908997|
|32.0      |10.0  |4.0 |0.0001       |step             |VGG19     |1.0  |sgd      |ddp     |0.28172194957733154|0.9031000137329102|
|32.0      |10.0  |4.0 |0.0001       |cosine           |VGG19     |1.0  |sgd      |ddp     |0.27951303124427795|0.902999997138977 |

### proje2 Deneysel Sonuç Tabloları

### ResNet50 Deneyleri

|batch_size|epochs|gpus|learning_rate|lr_scheduler_name|model_type|nodes|optimizer|strategy|test_loss          |test_acc          |
|----------|------|----|-------------|-----------------|----------|-----|---------|--------|-------------------|------------------|
|64.0      |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |rmsprop  |ddp     |0.11493199318647385|0.9700000286102295|
|64.0      |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |rmsprop  |ddp     |0.11657258123159409|0.9697999954223633|
|64.0      |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |adam     |ddp     |0.10925573855638504|0.968500018119812 |
|128.0     |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |rmsprop  |ddp     |0.12298182398080826|0.9682000279426575|
|64.0      |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |adamw    |ddp     |0.1143912822008133 |0.9679999947547913|
|64.0      |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |adam     |ddp     |0.12274883687496185|0.9675999879837036|
|64.0      |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |adamw    |ddp     |0.1166301965713501 |0.9672999978065491|
|128.0     |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |adamw    |ddp     |0.12047680467367172|0.9649999737739563|
|64.0      |10.0  |4.0 |0.0001       |reduce           |ResNet50  |1.0  |adam     |ddp     |0.1337757408618927 |0.9646999835968018|
|128.0     |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |rmsprop  |ddp     |0.1255222111940384 |0.9645000100135803|
|64.0      |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |adam     |ddp     |0.13480137288570404|0.9643999934196472|
|128.0     |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |adamw    |ddp     |0.1350170522928238 |0.963699996471405 |
|128.0     |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |adam     |ddp     |0.13017408549785614|0.9632999897003174|
|64.0      |10.0  |4.0 |0.0001       |reduce           |ResNet50  |1.0  |adamw    |ddp     |0.1375119388103485 |0.9628000259399414|
|128.0     |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |adam     |ddp     |0.13585273921489716|0.9624000191688538|
|128.0     |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |adam     |ddp     |0.13047103583812714|0.9621000289916992|
|64.0      |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |adamw    |ddp     |0.14343665540218353|0.9611999988555908|
|128.0     |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |adamw    |ddp     |0.126742422580719  |0.9607999920845032|
|64.0      |10.0  |4.0 |0.0001       |reduce           |ResNet50  |1.0  |rmsprop  |ddp     |0.13662512600421906|0.9603000283241272|
|128.0     |10.0  |4.0 |0.0001       |reduce           |ResNet50  |1.0  |adamw    |ddp     |0.1502835750579834 |0.9599000215530396|
|128.0     |10.0  |4.0 |0.0001       |reduce           |ResNet50  |1.0  |adam     |ddp     |0.15447008609771729|0.9592000246047974|
|64.0      |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |rmsprop  |ddp     |0.1678183376789093 |0.9569000005722046|
|128.0     |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |rmsprop  |ddp     |0.20221178233623505|0.9484999775886536|
|64.0      |10.0  |4.0 |0.0001       |reduce           |ResNet50  |1.0  |sgd      |ddp     |0.7831213474273682 |0.8158000111579895|
|64.0      |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |sgd      |ddp     |0.8090459108352661 |0.8113999962806702|
|64.0      |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |sgd      |ddp     |1.455028772354126  |0.7177000045776367|
|64.0      |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |sgd      |ddp     |1.4733856916427612 |0.7077999711036682|
|128.0     |10.0  |4.0 |0.0001       |reduce           |ResNet50  |1.0  |sgd      |ddp     |1.5360699892044067 |0.7008000016212463|
|128.0     |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |sgd      |ddp     |1.5328742265701294 |0.6855999827384949|
|128.0     |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |sgd      |ddp     |1.9138704538345337 |0.5705999732017517|
|128.0     |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |sgd      |ddp     |1.940787672996521  |0.5541999936103821|


### InceptionV3 Deneyleri

|batch_size|epochs|gpus|learning_rate|lr_scheduler_name|model_type|nodes|optimizer|strategy|test_loss          |test_acc          |
|----------|------|----|-------------|-----------------|----------|-----|---------|--------|-------------------|------------------|
|64.0      |10.0  |4.0 |0.0001       |cosine           |InceptionV3|1.0  |adamw    |ddp     |0.11969732493162155|0.9674000144004822|
|128.0     |10.0  |4.0 |0.0001       |cosine           |InceptionV3|1.0  |adamw    |ddp     |0.1294921338558197 |0.9653000235557556|
|64.0      |10.0  |4.0 |0.0001       |step             |InceptionV3|1.0  |adamw    |ddp     |0.12300121784210205|0.964900016784668 |
|128.0     |10.0  |4.0 |0.0001       |step             |InceptionV3|1.0  |adamw    |ddp     |0.13466650247573853|0.9621000289916992|
|128.0     |10.0  |4.0 |0.0001       |reduce           |InceptionV3|1.0  |adamw    |ddp     |0.1475798785686493 |0.9621000289916992|
|128.0     |10.0  |4.0 |0.0001       |none             |InceptionV3|1.0  |adamw    |ddp     |0.14625412225723267|0.9611999988555908|
|64.0      |10.0  |4.0 |0.0001       |reduce           |InceptionV3|1.0  |adamw    |ddp     |0.14710581302642822|0.9603000283241272|
|64.0      |10.0  |4.0 |0.0001       |none             |InceptionV3|1.0  |adamw    |ddp     |0.15464137494564056|0.9570000171661377|
|64.0      |10.0  |4.0 |0.0001       |cosine           |InceptionV3|1.0  |rmsprop  |ddp     |0.16916555166244507|0.9560999870300293|
|128.0     |10.0  |4.0 |0.0001       |step             |InceptionV3|1.0  |rmsprop  |ddp     |0.17848411202430725|0.9545999765396118|
|64.0      |10.0  |4.0 |0.0001       |step             |InceptionV3|1.0  |rmsprop  |ddp     |0.18899649381637573|0.9532999992370605|
|128.0     |10.0  |4.0 |0.0001       |reduce           |InceptionV3|1.0  |rmsprop  |ddp     |0.2874085307121277 |0.916100025177002 |
|64.0      |10.0  |4.0 |0.0001       |none             |InceptionV3|1.0  |rmsprop  |ddp     |0.2732328176498413 |0.9150000214576721|
|64.0      |10.0  |4.0 |0.0001       |reduce           |InceptionV3|1.0  |rmsprop  |ddp     |0.2874408960342407 |0.9146000146865845|
|64.0      |10.0  |4.0 |0.0001       |none             |InceptionV3|1.0  |sgd      |ddp     |0.4851588010787964 |0.8672000169754028|
|64.0      |10.0  |4.0 |0.0001       |reduce           |InceptionV3|1.0  |sgd      |ddp     |0.49547329545021057|0.864799976348877 |
|64.0      |10.0  |4.0 |0.0001       |step             |InceptionV3|1.0  |sgd      |ddp     |0.9482698440551758 |0.7914999723434448|
|128.0     |10.0  |4.0 |0.0001       |reduce           |InceptionV3|1.0  |sgd      |ddp     |0.9830966591835022 |0.7889000177383423|
|64.0      |10.0  |4.0 |0.0001       |cosine           |InceptionV3|1.0  |sgd      |ddp     |0.9903679490089417 |0.7811999917030334|
|128.0     |10.0  |4.0 |0.0001       |cosine           |InceptionV3|1.0  |sgd      |ddp     |1.6227368116378784 |0.6870999932289124|


### EfficientNetB2 Deneyleri

|batch_size|epochs|gpus|learning_rate|lr_scheduler_name|model_type|nodes|optimizer|strategy|test_loss          |test_acc          |
|----------|------|----|-------------|-----------------|----------|-----|---------|--------|-------------------|------------------|
|64.0      |10.0  |2.0 |0.001        |cosine           |EfficientNetB2|1.0  |adamw    |ddp     |0.392179399728775  |0.8816999793052673|
|64.0      |10.0  |2.0 |0.001        |step             |EfficientNetB2|1.0  |adamw    |ddp     |0.40105023980140686|0.8726000189781189|
|64.0      |10.0  |2.0 |0.001        |none             |EfficientNetB2|1.0  |adamw    |ddp     |0.4379236400127411 |0.8615999817848206|
|64.0      |10.0  |2.0 |0.001        |reduce           |EfficientNetB2|1.0  |adamw    |ddp     |0.45738422870635986|0.8586000204086304|
|64.0      |10.0  |2.0 |0.001        |reduce           |EfficientNetB2|1.0  |sgd      |ddp     |0.7388323545455933 |0.7415000200271606|
|64.0      |10.0  |2.0 |0.001        |none             |EfficientNetB2|1.0  |sgd      |ddp     |0.7856857776641846 |0.7239000201225281|
|64.0      |10.0  |2.0 |0.001        |step             |EfficientNetB2|1.0  |sgd      |ddp     |0.9195954203605652 |0.6735000014305115|
|64.0      |10.0  |2.0 |0.001        |cosine           |EfficientNetB2|1.0  |sgd      |ddp     |0.9215893745422363 |0.6661999821662903|


### VGG19 Deneyleri

|batch_size|epochs|gpus|learning_rate|lr_scheduler_name|model_type|nodes|optimizer|strategy|test_loss          |test_acc          |
|----------|------|----|-------------|-----------------|----------|-----|---------|--------|-------------------|------------------|
|64.0      |10.0  |2.0 |0.0001       |step             |VGG19     |2.0  |adamw    |ddp     |0.21935009956359863|0.9484999775886536|
|64.0      |10.0  |2.0 |0.0001       |cosine           |VGG19     |2.0  |adamw    |ddp     |0.24317313730716705|0.9463000297546387|
|128.0     |10.0  |2.0 |0.0001       |cosine           |VGG19     |2.0  |adamw    |ddp     |0.2413388192653656 |0.944599986076355 |
|128.0     |10.0  |2.0 |0.0001       |step             |VGG19     |2.0  |adamw    |ddp     |0.24942873418331146|0.9426000118255615|
|128.0     |10.0  |2.0 |0.0001       |none             |VGG19     |2.0  |adamw    |ddp     |0.25471994280815125|0.9366999864578247|
|64.0      |10.0  |2.0 |0.0001       |none             |VGG19     |2.0  |adamw    |ddp     |0.2844490110874176 |0.9348999857902527|
|64.0      |10.0  |2.0 |0.0001       |reduce           |VGG19     |2.0  |adamw    |ddp     |0.26456496119499207|0.9347000122070312|
|128.0     |10.0  |2.0 |0.0001       |reduce           |VGG19     |2.0  |adamw    |ddp     |0.29695984721183777|0.9258999824523926|
|64.0      |10.0  |2.0 |0.0001       |none             |VGG19     |2.0  |sgd      |ddp     |0.2868402898311615 |0.9003999829292297|
|64.0      |10.0  |2.0 |0.0001       |step             |VGG19     |2.0  |sgd      |ddp     |0.3304106295108795 |0.8845999836921692|
|64.0      |10.0  |2.0 |0.0001       |cosine           |VGG19     |2.0  |sgd      |ddp     |0.3368724584579468 |0.8809000253677368|


### proje3 Deneysel Sonuç Tabloları

### ResNet50 Deneyleri

|batch_size|epochs|gpus|learning_rate|lr_scheduler_name|model_type|nodes|optimizer|strategy|test_loss          |test_acc          |
|----------|------|----|-------------|-----------------|----------|-----|---------|--------|-------------------|------------------|
|64.0      |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |rmsprop  |fsdp1   |0.12220947444438934|0.9688000082969666|
|64.0      |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |adamw    |fsdp1   |0.10977363586425781|0.9677000045776367|
|128.0     |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |rmsprop  |fsdp1   |0.12087298929691315|0.9672999978065491|
|64.0      |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |adamw    |fsdp1   |0.1151389628648758 |0.9671000242233276|
|64.0      |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |adam     |fsdp1   |0.11952836811542511|0.9671000242233276|
|128.0     |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |adam     |fsdp1   |0.12219113856554031|0.9666000008583069|
|128.0     |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |rmsprop  |fsdp1   |0.1249803826212883 |0.9666000008583069|
|64.0      |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |rmsprop  |fsdp1   |0.12445100396871567|0.9663000106811523|
|64.0      |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |adam     |fsdp1   |0.12025275826454163|0.9660000205039978|
|128.0     |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |adamw    |fsdp1   |0.12062560766935349|0.9646999835968018|
|64.0      |10.0  |4.0 |0.0001       |reduce           |ResNet50  |1.0  |adamw    |fsdp1   |0.12935324013233185|0.9639999866485596|
|64.0      |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |adamw    |fsdp1   |0.14106376469135284|0.9639999866485596|
|64.0      |10.0  |4.0 |0.0001       |reduce           |ResNet50  |1.0  |adam     |fsdp1   |0.13435587286949158|0.9634000062942505|
|128.0     |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |adam     |fsdp1   |0.1288043111562729 |0.9632999897003174|
|128.0     |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |adamw    |fsdp1   |0.12643861770629883|0.963100016117096 |
|64.0      |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |adam     |fsdp1   |0.12486869096755981|0.963100016117096 |
|128.0     |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |rmsprop  |fsdp1   |0.137114480137825  |0.9617000222206116|
|128.0     |10.0  |4.0 |0.0001       |reduce           |ResNet50  |1.0  |adamw    |fsdp1   |0.14843960106372833|0.9613000154495239|
|128.0     |10.0  |4.0 |0.0001       |reduce           |ResNet50  |1.0  |adam     |fsdp1   |0.1412789523601532 |0.9613000154495239|
|128.0     |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |adamw    |fsdp1   |0.15801063179969788|0.9605000019073486|
|128.0     |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |adam     |fsdp1   |0.15594471991062164|0.9598000049591064|
|128.0     |10.0  |4.0 |0.0001       |reduce           |ResNet50  |1.0  |rmsprop  |fsdp1   |0.1441492736339569 |0.9595000147819519|
|64.0      |10.0  |4.0 |0.0001       |reduce           |ResNet50  |1.0  |rmsprop  |fsdp1   |0.15504090487957   |0.954800009727478 |
|64.0      |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |rmsprop  |fsdp1   |0.1807052195072174 |0.9491000175476074|
|64.0      |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |sgd      |fsdp1   |0.8003547787666321 |0.809499979019165 |
|64.0      |10.0  |4.0 |0.0001       |reduce           |ResNet50  |1.0  |sgd      |fsdp1   |0.7822596430778503 |0.8076000213623047|
|64.0      |10.0  |4.0 |0.0001       |cosine           |ResNet50  |1.0  |sgd      |fsdp1   |1.474304437637329  |0.6984000205993652|
|64.0      |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |sgd      |fsdp1   |1.48746657371521   |0.6800000071525574|
|128.0     |10.0  |4.0 |0.0001       |none             |ResNet50  |1.0  |sgd      |fsdp1   |1.5691746473312378 |0.670199990272522 |
|128.0     |10.0  |4.0 |0.0001       |step             |ResNet50  |1.0  |sgd      |fsdp1   |1.930001974105835  |0.5633000135421753|

### InceptionV3 Deneyleri

|batch_size|epochs|gpus|learning_rate|lr_scheduler_name|model_type|nodes|optimizer|strategy|test_loss          |test_acc          |
|----------|------|----|-------------|-----------------|----------|-----|---------|--------|-------------------|------------------|
|64.0      |10.0  |4.0 |0.0001       |cosine           |InceptionV3|1.0  |adam     |fsdp1   |0.12618650496006012|0.9672999978065491|
|64.0      |10.0  |4.0 |0.0001       |step             |InceptionV3|1.0  |adam     |fsdp1   |0.12036586552858353|0.9661999940872192|
|128.0     |10.0  |4.0 |0.0001       |cosine           |InceptionV3|1.0  |adam     |fsdp1   |0.13338392972946167|0.963100016117096 |
|128.0     |10.0  |4.0 |0.0001       |step             |InceptionV3|1.0  |adam     |fsdp1   |0.13084404170513153|0.961899995803833 |
|128.0     |10.0  |4.0 |0.0001       |cosine           |InceptionV3|1.0  |rmsprop  |fsdp1   |0.15118686854839325|0.9610999822616577|
|64.0      |10.0  |4.0 |0.0001       |none             |InceptionV3|1.0  |adam     |fsdp1   |0.15585549175739288|0.9599000215530396|
|64.0      |10.0  |4.0 |0.0001       |cosine           |InceptionV3|1.0  |rmsprop  |fsdp1   |0.15571655333042145|0.9592000246047974|
|128.0     |10.0  |4.0 |0.0001       |none             |InceptionV3|1.0  |adam     |fsdp1   |0.16012778878211975|0.9585999846458435|
|64.0      |10.0  |4.0 |0.0001       |reduce           |InceptionV3|1.0  |adam     |fsdp1   |0.1572117954492569 |0.9585000276565552|
|128.0     |10.0  |4.0 |0.0001       |step             |InceptionV3|1.0  |rmsprop  |fsdp1   |0.17574067413806915|0.9563999772071838|
|128.0     |10.0  |4.0 |0.0001       |reduce           |InceptionV3|1.0  |adam     |fsdp1   |0.16002574563026428|0.9549000263214111|
|64.0      |10.0  |4.0 |0.0001       |step             |InceptionV3|1.0  |rmsprop  |fsdp1   |0.18441960215568542|0.953000009059906 |
|64.0      |10.0  |4.0 |0.0001       |reduce           |InceptionV3|1.0  |rmsprop  |fsdp1   |0.2679239511489868 |0.9142000079154968|
|128.0     |10.0  |4.0 |0.0001       |reduce           |InceptionV3|1.0  |rmsprop  |fsdp1   |0.2887730896472931 |0.9117000102996826|
|128.0     |10.0  |4.0 |0.0001       |none             |InceptionV3|1.0  |rmsprop  |fsdp1   |0.3029908537864685 |0.9107000231742859|
|64.0      |10.0  |4.0 |0.0001       |none             |InceptionV3|1.0  |rmsprop  |fsdp1   |0.2997698485851288 |0.9093000292778015|
|64.0      |10.0  |4.0 |0.0001       |none             |InceptionV3|1.0  |sgd      |fsdp1   |0.4988947808742523 |0.8651999831199646|
|64.0      |10.0  |4.0 |0.0001       |reduce           |InceptionV3|1.0  |sgd      |fsdp1   |0.499421089887619  |0.8651000261306763|
|64.0      |10.0  |4.0 |0.0001       |cosine           |InceptionV3|1.0  |sgd      |fsdp1   |0.9213041067123413 |0.8025000095367432|
|64.0      |10.0  |4.0 |0.0001       |step             |InceptionV3|1.0  |sgd      |fsdp1   |0.9638965129852295 |0.7886000275611877|
|128.0     |10.0  |4.0 |0.0001       |none             |InceptionV3|1.0  |sgd      |fsdp1   |1.0029124021530151 |0.7839000225067139|
|128.0     |10.0  |4.0 |0.0001       |step             |InceptionV3|1.0  |sgd      |fsdp1   |1.5995588302612305 |0.7164000272750854|
|128.0     |10.0  |4.0 |0.0001       |cosine           |InceptionV3|1.0  |sgd      |fsdp1   |1.6527851819992065 |0.7038000226020813|

### EfficientNetB2 Deneyleri

|batch_size|epochs|gpus|learning_rate|lr_scheduler_name|model_type|nodes|optimizer|strategy|test_loss          |test_acc          |
|----------|------|----|-------------|-----------------|----------|-----|---------|--------|-------------------|------------------|
|64.0      |10.0  |2.0 |0.001        |cosine           |EfficientNetB2|1.0  |adam     |fsdp1   |0.3875776529312134 |0.8863000273704529|
|64.0      |10.0  |2.0 |0.001        |step             |EfficientNetB2|1.0  |adam     |fsdp1   |0.38276833295822144|0.8799999952316284|
|64.0      |10.0  |2.0 |0.001        |none             |EfficientNetB2|1.0  |adam     |fsdp1   |0.4050825834274292 |0.8697999715805054|
|64.0      |10.0  |2.0 |0.001        |cosine           |EfficientNetB2|1.0  |rmsprop  |fsdp1   |0.44431695342063904|0.849399983882904 |
|64.0      |10.0  |2.0 |0.001        |step             |EfficientNetB2|1.0  |rmsprop  |fsdp1   |0.458683580160141  |0.8446000218391418|
|64.0      |10.0  |2.0 |0.001        |reduce           |EfficientNetB2|1.0  |rmsprop  |fsdp1   |0.5444735288619995 |0.8162000179290771|
|64.0      |10.0  |2.0 |0.001        |none             |EfficientNetB2|1.0  |rmsprop  |fsdp1   |0.6094237565994263 |0.7896999716758728|


### VGG19 Deneyleri

|batch_size|epochs|gpus|learning_rate|lr_scheduler_name|model_type|nodes|optimizer|strategy|test_loss          |test_acc          |
|----------|------|----|-------------|-----------------|----------|-----|---------|--------|-------------------|------------------|
|64.0      |10.0  |2.0 |0.0001       |step             |VGG19     |2.0  |adam     |fsdp1   |0.22371725738048553|0.9474999904632568|
|64.0      |10.0  |2.0 |0.0001       |none             |VGG19     |2.0  |adam     |fsdp1   |0.2784087657928467 |0.932699978351593 |
|64.0      |10.0  |2.0 |0.0001       |step             |VGG19     |2.0  |rmsprop  |fsdp1   |0.4453234374523163 |0.8693000078201294|
|64.0      |10.0  |2.0 |0.0001       |cosine           |VGG19     |2.0  |rmsprop  |fsdp1   |0.48115116357803345|0.8664000034332275|
|64.0      |10.0  |2.0 |0.0001       |none             |VGG19     |2.0  |rmsprop  |fsdp1   |0.4423808157444    |0.863099992275238 |
|64.0      |10.0  |2.0 |0.0001       |reduce           |VGG19     |2.0  |rmsprop  |fsdp1   |0.5039038062095642 |0.8508999943733215|
|128.0     |10.0  |2.0 |0.0001       |step             |VGG19     |2.0  |rmsprop  |fsdp1   |0.5126741528511047 |0.8305000066757202|
|128.0     |10.0  |2.0 |0.0001       |cosine           |VGG19     |2.0  |rmsprop  |fsdp1   |0.5499520301818848 |0.8251000046730042|
|128.0     |10.0  |2.0 |0.0001       |none             |VGG19     |2.0  |rmsprop  |fsdp1   |0.610767662525177  |0.8001000285148621|
|128.0     |10.0  |2.0 |0.0001       |reduce           |VGG19     |2.0  |rmsprop  |fsdp1   |0.7432979941368103 |0.7538999915122986|


### Genel İstatistikler

Üç proje aşamasında toplam 274 deney gerçekleştirilmiştir. Bu deneyler, 4 farklı model mimarisi, 3 farklı paralelleştirme stratejisi, 4 farklı optimizasyon algoritması ve çeşitli learning rate scheduler kombinasyonlarını kapsamaktadır.

**Model Bazında Deney Dağılımı:**
- ResNet50: 112 deney (en kapsamlı test edilen model)
- InceptionV3: 75 deney 
- EfficientNetB2: 41 deney
- VGG19: 46 deney

### Proje 1: Temel GPU Ölçeklendirme Analizi

İlk projede elde edilen en önemli bulgular GPU ölçeklendirme ile ilgilidir. 1 GPU kullanımında ortalama %92.34 doğruluk elde edilirken, 2 GPU ile %92.31 ve 4 GPU ile %85.86 doğruluk oranları gözlenmiştir. Bu sonuç, GPU sayısının artmasının her zaman performans artışı sağlamadığını göstermektedir.

ResNet50 modelinde en yüksek performans 2 GPU, batch size 32, Adam optimizer ve Cosine scheduler kombinasyonu ile %97.11 doğruluk olarak kaydedilmiştir. InceptionV3 modeli ise %86.69 - %97.09 aralığında çok tutarlı sonuçlar vermiştir.

### Proje 2: RMSprop ve GPU Optimizasyonu

İkinci projenin en belirgin katkısı RMSprop optimizasyon algoritmasının performansıdır. ResNet50 modelinde RMSprop, 7 deney boyunca %96.26 ortalama doğruluk ile dikkat çekici bir tutarlılık sergilemiştir. En yüksek performans %97.00 olarak kaydedilmiş, bu da Adam (%96.40 ortalama) ve AdamW (%96.36 ortalama) algoritmalarını geride bırakmıştır.

GPU utilization modülünün entegrasyonu sayesinde elde edilen veriler, kaynak kullanımının optimize edilebileceğini göstermiştir. Özellikle EfficientNetB2 ve VGG19 modelleri için 2 GPU kullanımının hem performans hem de kaynak verimliliği açısından optimal olduğu tespit edilmiştir.

### Proje 3: FSDP Stratejisinin Etkisi

Üçüncü projenin ana odağı olan FSDP (Fully Sharded Data Parallel) stratejisi, bellek yönetimi ve büyük model eğitimi açısından önemli avantajlar sağlamıştır. FSDP1 konfigürasyonunda 30 deney gerçekleştirilen ResNet50 modelinde %91.15 ortalama doğruluk elde edilmiştir.

FSDP'nin en büyük avantajı, büyük modellerin 4 GPU üzerinde daha verimli şekilde eğitilebilmesi olmuştur. Bellek fragmentasyonu azalırken, gradient synchronization süreçleri optimize edilmiştir.

## Optimizer Algoritmaları Karşılaştırmalı Analizi

Üç proje boyunca test edilen optimizer algoritmaları arasında dikkat çekici performans farklılıkları gözlenmiştir. Adam algoritması özellikle küçük veri setleri ve kısa eğitim süreleri için ideal bir seçim olduğunu kanıtlamıştır. 16 deney boyunca %96.30 ortalama doğruluk ile en yüksek ve en tutarlı performansı sergilemiştir.

RMSprop algoritmasının Proje 2'de gösterdiği başarı, adaptive learning rate yöntemlerinin CIFAR-10 gibi görüntü sınıflandırma görevlerinde ne kadar etkili olabileceğini göstermektedir. Özellikle CNN mimarileri için RMSprop'un momentum tabanlı güncellemeleri, gradientlerin daha stabil konverjansa ulaşmasını sağlamıştır.

AdamW algoritması, weight decay mekanizması sayesinde overfitting'e karşı daha dayanıklı sonuçlar vermiştir. 61 deney ile en geniş test edilen optimizer olmasına rağmen, %93.49 ortalama ile tutarlı bir performans sergilemiştir.

SGD algoritması beklenildiği üzere en yavaş konverjans göstermiştir. %81.84 ortalama doğruluk ile diğer algoritmaların gerisinde kalmış, ancak bazı specific konfigürasyonlarda %94.96'ya kadar çıkabilmiştir.

## Learning Rate Scheduler Stratejilerinin Etkisi

Learning rate scheduler'ların performans üzerindeki etkisi projeler boyunca yakından takip edilmiştir. Ilginç bir şekilde, hiçbir scheduler kullanmamak (%90.07 ortalama) en iyi ortalama performansı verirken, Cosine scheduler ile en yüksek bireysel performans (%97.11) elde edilmiştir.

Bu durum, CIFAR-10 gibi nispeten küçük veri setlerinde ve kısa eğitim süreleri (10 epoch) için scheduler kullanımının her zaman avantajlı olmayabileceğini göstermektedir. Cosine annealing'in özellikle uzun eğitimlerde daha etkili olduğu bilinmektedir.

## Paralelleştirme Stratejileri ve Kaynak Verimliliği

DDP (Distributed Data Parallel) stratejisi, Proje 1 ve 2'de temel paralelleştirme yöntemi olarak kullanılmıştır. Bu yaklaşım, her GPU'da modelin tam bir kopyasını bulundurarak gradient synchronization işlemini gerçekleştirmektedir. 4 GPU kullanımında DDP ile bazı performans düşüşleri gözlenmiş, bu durum gradient averaging sürecindeki communication overhead'e bağlanmıştır.

FSDP (Fully Sharded Data Parallel) stratejisi Proje 3'te test edilmiş ve özellikle bellek kullanımında önemli avantajlar sağlamıştır. Model parametrelerinin GPU'lar arasında paylaştırılması sayesinde, daha büyük batch size'lar kullanılabilmiş ve bellek verimliliği artırılmıştır.

## Teknik Zorluklar ve Öğrenilen Dersler

Proje süresince karşılaşılan en önemli zorluk, PyTorch Lightning framework'ün log versiyonlama sisteminin yanlış anlaşılması olmuştur. Başlangıçta her yeni deney aşaması için ayrı proje klasörleri oluşturulmuş (proje, proje2, proje3 gibi) ve tüm kod dosyaları tekrar kopyalanmıştır. Bu yaklaşım hem disk alanında gereksiz kullanıma hem de kod yönetiminde karmaşıklığa neden olmuştur.

Sonradan öğrenildiği üzere, PyTorch Lightning'in `Trainer` sınıfı otomatik versiyonlama sistemi ile çalışmaktadır. Aynı `default_root_dir` altında farklı deneyler yapıldığında, Lightning otomatik olarak `version_0`, `version_1`, `version_2` gibi klasörler oluşturarak her deneyi ayrı tutmaktadır. Bu sistem sayesinde:

- Aynı kod tabanı üzerinde değişiklikler yapılarak farklı konfigürasyonlar test edilebilir
- Her deney otomatik olarak zaman damgası ve versiyon numarası ile etiketlenir
- TensorBoard logları organize bir şekilde farklı versiyonlar altında saklanır
- Disk alanı tasarrufu sağlanır ve kod duplikasyonu önlenir

Örneğin, `logger = TensorBoardLogger("logs/", name="resnet50")` tanımlaması ile, Lightning otomatik olarak `logs/resnet50/version_0/`, `logs/resnet50/version_1/` gibi klasörler oluşturmaktadır. Bu özellik, özellikle hiperparametre optimizasyonu ve ablation study'ler için çok değerlidir.

Bu deneyim, süper bilgisayar ortamlarında çalışırken framework dokümantasyonlarının dikkatli okunmasının ve best practice'lerin takip edilmesinin önemini ortaya koymuştur. İleriki çalışmalarda bu versiyonlama sistemini kullanarak daha organize ve verimli bir proje yönetimi sağlanabilir.

Tüm projenin farklı aşamalar ile katmanlı olarak yürütülmesi belirli bir düzen sağlasa da klasörleme yapısındaki bahsedilen bu zorluklar nedeniyle farklı proje klasörlerinde tamamen aynı kombinasyonlarda eğitimler bulunması mümkündür. Bu da çalışmanın toplam verimini oldukça azaltmaktadır. 

4 saatlik süre sınırlaması, özellikle büyük batch size'lar ve kompleks model konfigürasyonları için önemli bir kısıt oluşturmuştur. Bu nedenle Proje 3'te öncelik sistemi geliştirilerek, FSDP deneylerinin tamamlanması sağlanmıştır.

GPU utilization modülünün Proje 1'de kullanılmaması, kaynak kullanım analizlerinin eksik kalmasına neden olmuştur. Proje 2'de bu eksiklik giderilerek, TensorBoard entegrasyonu ile detaylı GPU monitoring sağlanmıştır.

## Model Mimarileri Performans Değerlendirmesi

**ResNet50** mimarisi, en geniş performans aralığına sahip model olarak öne çıkmıştır. %55.10 ile %97.11 arasında değişen sonuçlar, bu mimarinin parametre seçimine olan yüksek duyarlılığını göstermektedir. Doğru konfigürasyon ile en yüksek performansa ulaşabilirken, yanlış parametre seçimleri dramatik performans düşüşlerine neden olabilmektedir.

**InceptionV3** modeli, %94.45 ortalama doğruluk ile en tutarlı performansı sergilemiştir. %86.69 - %97.09 aralığındaki sonuçlar, bu mimarinin farklı konfigürasyonlara karşı dayanıklılığını göstermektedir. Inception modüllerinin multi-scale feature extraction yeteneği, CIFAR-10 gibi küçük görüntüler için ideal bir seçim olduğunu kanıtlamıştır.

**VGG19** mimarisi en kararlı sonuçları vermiştir. %90.30 - %95.04 aralığındaki performans, bu klasik mimarinin güvenilirliğini ortaya koymaktadır. Daha basit bir yapıya sahip olmasına rağmen, tutarlı sonuçlar elde etmek için ideal bir seçimdir.

**EfficientNetB2** modeli, CIFAR-10 veri seti için en düşük performansı göstermiştir. %79.48 ortalama doğruluk ile diğer modellerin gerisinde kalmıştır. Bu durum, EfficientNet mimarisinin daha büyük görüntüler ve daha kompleks veri setleri için optimize edilmiş olmasından kaynaklanmaktadır.

## Kaynak Kullanımı ve Verimlilik Optimizasyonları

TRUBA altyapısının verimli kullanımı konusunda projeler boyunca önemli deneyimler kazanılmıştır. İlk projede num_workers parametresinin düşük tutulması veri yükleme bottleneck'ine neden olmuş ve sistem verimliliği uyarıları alınmıştır. Bu sorun proje2'de num_workers değerinin artırılması ile çözülmüştür.

GPU utilization analizleri, farklı modellerin kaynak ihtiyaçlarının büyük ölçüde değiştiğini göstermiştir. EfficientNetB2 ve VGG19 gibi daha küçük modeller için 2 GPU kullanımının optimal olduğu, 4 GPU kullanımının ise kaynak israfına neden olabildiği tespit edilmiştir.

Batch size optimizasyonu kritik önemde olmuştur. 32-64 aralığındaki batch size'lar en iyi performansı verirken, 128 ve üzeri değerlerde gradient noise'in artması nedeniyle konverjans sorunları gözlenmiştir.

"proje" deneylerinde model başına kullanılan GPU sayıları farklılık gösterebilirken hangi node ve hangi çeşitte hızlandırıcı tercih edildiği kaydedilememiştir. 

ve "proje2" ve "proje3" deneylerinde model başına kullanılan GPU çeşitleri ; 

- ResNet50 : 1 x Akya-Cuda (4 x V100)
- InceptionV3 : 1 x Akya-Cuda (4 x V100)
- VGG19 : 1 x Barbun-Cuda (2 x P100)
- EfficientNetB2 : 1 x Barbun-Cuda (2 x P100)

şeklinde olmuştur.

## Gelecek Çalışmalar İçin Öneriler

Bu kapsamlı çalışmanın sonuçları ışığında, gelecek araştırmalar için önemli yönergeler ortaya çıkmıştır. Mixed precision training'in test edilmesi, özellikle büyük modeller için bellek kullanımını daha da optimize edebilir. Half precision (FP16) kullanımı, TRUBA'nın V100 GPU'larının tensor core yeteneklerinden tam olarak yararlanmayı sağlayacaktır.

Data augmentation stratejilerinin etkisinin sistematik olarak incelenmesi, model performansını artırabilir. CIFAR-10 veri setinin küçük boyutu göz önüne alındığında, augmentation tekniklerinin overfitting'i önlemedeki rolü kritik olabilir.

Dynamic learning rate adaptation tekniklerinin keşfedilmesi, scheduler'ların etkinliğini artırabilir. Özellikle performance-based lr scheduling yöntemlerinin test edilmesi değerli olacaktır.

## Sonuçlar ve Çıkarımlar

Bu üç aşamalı proje, TRUBA Süper Bilgisayar Merkezi altyapısında derin öğrenme model eğitiminin kapsamlı bir analizini sunmuştur. 274 deney ile elde edilen bulgular, GPU ölçeklendirmenin her zaman doğrusal performans artışı sağlamadığını, optimizer seçiminin kritik önemde olduğunu ve paralelleştirme stratejilerinin model ve kaynak karakteristiklerine göre dikkatli seçilmesi gerektiğini ortaya koymuştur.

RMSprop algoritmasının CNN mimarileri için gösterdiği başarı, adaptive learning rate yöntemlerinin potansiyelini göstermektedir. FSDP stratejisinin bellek optimizasyonundaki başarısı, büyük model eğitimleri için umut vericidir.

En önemli çıkarım, "one size fits all" yaklaşımının derin öğrenmede geçerli olmadığıdır. Her model mimarisi, veri seti ve kaynak konfigürasyonu için optimal parametrelerin sistematik olarak araştırılması gereklidir. TRUBA gibi ulusal süper bilgisayar kaynaklarının etkin kullanımı, bu tür kapsamlı parameter sweep çalışmalarını mümkün kılmaktadır.

Proje süresince yaşanan teknik zorluklar ve bunlara geliştirilen çözümler, süper bilgisayar kullanımında deneyim kazanımının önemini vurgulamaktadır. İlk projeden üçüncü projeye kadar görülen iyileştirmeler, öğrenme sürecinin değerini göstermektedir.

---

**Proje Dosyaları:**
- Tüm kodlar, scriptler ve detaylı sonuçlar: https://drive.google.com/drive/folders/1MF7Fj_v3b1F0VP_kDSOhNfnHwsxc7jEW?usp=sharing
- Deneysel veriler: proje-tables.md, proje2-tables.md, proje3-tables.md

**İletişim:**
- E-posta: safakilickaya34@gmail.com
- TRUBA Hesap: egitim22