"""Curated Turkish retrieval corpus: 100 passages and one question each.

All text is original, written for tr-eval-bench from widely known general
knowledge (geography, history, science, technology, nature, culture,
literature, sport, language).  No copyrighted corpus was copied, no personal
data is present, no real person names appear (people are described by role or
omitted), and no official statistic, tax, rate or legal value is quoted.
The dataset is released under the MIT license together with the project.

Each entry: (passage_id, domain, title, passage_text, question)
"""

from __future__ import annotations

PASSAGES: list[tuple[str, str, str, str, str]] = [
    (
        "p001",
        "cografya",
        "Anadolu Yarımadası",
        "Anadolu, kuzeyde Karadeniz, güneyde Akdeniz ile çevrili büyük bir yarımadadır. "
        "Batısında Ege Denizi, kuzeybatısında Marmara Denizi yer alır.",
        "Anadolu Yarımadası'nı çevreleyen denizler hangileridir?",
    ),
    (
        "p002",
        "cografya",
        "Toros Dağları",
        "Toros Dağları, Akdeniz kıyısına paralel uzanan ve Güney Anadolu'yu boydan boya "
        "geçen bir dağ sırasıdır.",
        "Güney Anadolu'da Akdeniz'e paralel uzanan dağ sırasının adı nedir?",
    ),
    (
        "p003",
        "cografya",
        "Van Gölü",
        "Van Gölü, Doğu Anadolu'da yer alan, sodalı suyuyla bilinen büyük bir kapalı havza "
        "gölüdür.",
        "Sodalı suyuyla bilinen Doğu Anadolu gölü hangisidir?",
    ),
    (
        "p004",
        "cografya",
        "Kızılırmak",
        "Kızılırmak, İç Anadolu'dan doğup Karadeniz'e ulaşan ve tamamen Türkiye sınırları "
        "içinde kalan en uzun akarsudur.",
        "Tamamen Türkiye sınırlarında kalan en uzun akarsu hangisidir?",
    ),
    (
        "p005",
        "cografya",
        "Boğazlar",
        "İstanbul ve Çanakkale boğazları, Karadeniz'i Ege ve Akdeniz'e bağlayan dar su "
        "geçitleridir.",
        "Karadeniz'i Ege Denizi'ne bağlayan boğazlar hangileridir?",
    ),
    (
        "p006",
        "cografya",
        "Kapadokya",
        "Kapadokya, volkanik tüflerin aşınmasıyla oluşan peri bacaları ve kaya kiliseleriyle "
        "tanınan bir bölgedir.",
        "Peri bacalarıyla ünlü volkanik bölge neresidir?",
    ),
    (
        "p007",
        "cografya",
        "Pamukkale Travertenleri",
        "Pamukkale'deki beyaz travertenler, kalsiyum karbonat yüklü sıcak suların zamanla "
        "çökelmesiyle oluşmuştur.",
        "Pamukkale travertenleri hangi kimyasal süreçle oluşur?",
    ),
    (
        "p008",
        "cografya",
        "Ege Kıyıları",
        "Ege kıyıları, dağların denize dik uzanması sayesinde girintili çıkıntılı koylar ve "
        "doğal limanlar barındırır.",
        "Ege kıyılarında doğal limanların çok olmasının nedeni nedir?",
    ),
    (
        "p009",
        "cografya",
        "Tuz Gölü",
        "Tuz Gölü, İç Anadolu'da yer alan sığ bir göldür; çevresindeki tuzlalarda göl suyundan "
        "tuz elde edilir.",
        "İç Anadolu'daki sığ tuz gölünün adı nedir?",
    ),
    (
        "p010",
        "cografya",
        "Fırat Nehri",
        "Fırat Nehri, Doğu Anadolu'dan doğup Suriye ve Irak üzerinden Basra Körfezi'ne ulaşan "
        "uzun bir nehirdir.",
        "Doğu Anadolu'dan doğup Basra Körfezi'ne dökülen nehir hangisidir?",
    ),
    (
        "p011",
        "tarih",
        "İstanbul'un Fethi",
        "İstanbul, 1453 yılında Osmanlı ordusu tarafından fethedildi ve imparatorluğun "
        "başkenti oldu.",
        "İstanbul hangi yılda Osmanlı ordusu tarafından fethedildi?",
    ),
    (
        "p012",
        "tarih",
        "Cumhuriyetin İlanı",
        "Türkiye Cumhuriyeti, 29 Ekim 1923'te TBMM'nin kararıyla ilan edildi.",
        "Cumhuriyet hangi tarihte ilan edildi?",
    ),
    (
        "p013",
        "tarih",
        "Göbeklitepe",
        "Göbeklitepe, Şanlıurfa yakınlarında bulunan ve dünyanın bilinen en eski tapınak "
        "alanlarından biri kabul edilen kazı alanıdır.",
        "Dünyanın en eski tapınak alanlarından biri sayılan kazı alanı nerededir?",
    ),
    (
        "p014",
        "tarih",
        "Truva",
        "Truva kenti, Çanakkale Boğazı'nın güneyinde yer alır ve İlyada destanına konu olmuştur.",
        "İlyada destanına konu olan antik kent hangisidir?",
    ),
    (
        "p015",
        "tarih",
        "Anadolu Selçuklu Devleti",
        "Anadolu Selçuklu Devleti'nin başkenti Konya'ydı; şehir bu dönemde önemli bir "
        "kültür merkeziydi.",
        "Anadolu Selçuklu Devleti'nin başkenti neresiydi?",
    ),
    (
        "p016",
        "tarih",
        "Kültepe",
        "Anadolu'da yazının ilk kullanıldığı yerlerden biri, Asurlu tüccarların kurduğu "
        "Kültepe karumudur.",
        "Anadolu'da yazının ilk kullanıldığı yerleşimlerden biri hangisidir?",
    ),
    (
        "p017",
        "tarih",
        "Efes Antik Kenti",
        "Efes Antik Kenti, Artemis Tapınağı ve büyük bir kütüphanesiyle bilinen, İzmir "
        "yakınlarındaki bir liman kentidir.",
        "Artemis Tapınağı hangi antik kentte bulunur?",
    ),
    (
        "p018",
        "tarih",
        "Kurtuluş Savaşı",
        "Kurtuluş Savaşı, 19 Mayıs 1919'da Samsun'a çıkışla başlatılan bağımsızlık mücadelesidir.",
        "Kurtuluş Savaşı hangi olayla başlatılmıştır?",
    ),
    (
        "p019",
        "tarih",
        "Osmanlı Başkentleri",
        "Osmanlı Devleti'nin başkentleri sırasıyla Bursa, Edirne ve İstanbul olmuştur.",
        "Osmanlı Devleti'nin İstanbul'dan önceki başkentleri hangileriydi?",
    ),
    (
        "p020",
        "tarih",
        "Çatalhöyük",
        "Çatalhöyük, Konya ovasında yer alan ve Neolitik döneme ait ev kalıntılarıyla ünlü bir "
        "höyüktür.",
        "Neolitik dönem evleriyle ünlü Konya'daki höyük hangisidir?",
    ),
    (
        "p021",
        "bilim",
        "Kaynama Noktası",
        "Deniz seviyesinde saf su 100 santigrat derecede kaynar; yükseldikçe azalan hava "
        "basıncı nedeniyle kaynama noktası düşer.",
        "Yükseklik arttıkça suyun kaynama noktası nasıl değişir?",
    ),
    (
        "p022",
        "bilim",
        "Fotosentez",
        "Fotosentez, bitkilerin güneş ışığını kullanarak karbondioksit ve sudan şeker "
        "ürettiği süreçtir.",
        "Bitkiler fotosentezde hangi gazı kullanır?",
    ),
    (
        "p023",
        "bilim",
        "Yerçekimi",
        "Yerçekimi, kütlesi olan cisimlerin birbirini çekmesine neden olan temel kuvvettir.",
        "Cisimlerin birbirini çekmesine neden olan kuvvet nedir?",
    ),
    (
        "p024",
        "bilim",
        "Atom",
        "Atom, çekirdek ile çekirdeğin çevresinde dolanan elektronlardan oluşan maddenin temel "
        "yapı taşıdır.",
        "Atomun çekirdeğinin çevresinde dolanan parçacıklara ne denir?",
    ),
    (
        "p025",
        "bilim",
        "Periyodik Tablo",
        "Periyodik tablo, elementleri artan atom numarasına göre sıralar ve benzer "
        "özellikteki elementleri aynı grupta toplar.",
        "Periyodik tabloda elementler hangi özelliğe göre sıralanır?",
    ),
    (
        "p026",
        "bilim",
        "DNA",
        "DNA, canlıların kalıtsal bilgisini taşıyan ve çift sarmal yapıda bulunan bir moleküldür.",
        "Kalıtsal bilgiyi taşıyan çift sarmal molekül hangisidir?",
    ),
    (
        "p027",
        "bilim",
        "Buharlaşma",
        "Buharlaşma, sıvı hâldeki moleküllerin yüzeyden gaz hâline geçmesi olayıdır ve "
        "sıcaklık arttıkça hızlanır.",
        "Sıcaklık arttıkça buharlaşma nasıl etkilenir?",
    ),
    (
        "p028",
        "bilim",
        "Asitler ve Bazlar",
        "Asitler turnusol kâğıdını kırmızıya, bazlar maviye çevirir; ikisi tepkimeye "
        "girdiğinde nötrleşme olur.",
        "Asit ile baz tepkimeye girdiğinde hangi olay gerçekleşir?",
    ),
    (
        "p029",
        "bilim",
        "Işığın Dispersiyonu",
        "Beyaz ışık prizmadan geçerken renklerine ayrılır; bu olaya ışığın dispersiyonu denir.",
        "Beyaz ışığın prizmada renklere ayrılmasına ne denir?",
    ),
    (
        "p030",
        "bilim",
        "Hücre",
        "Hücre, canlıların en küçük yapı ve görev birimidir; prokaryot hücrelerde çekirdek "
        "zarı bulunmaz.",
        "Çekirdek zarı bulunmayan hücre tipi hangisidir?",
    ),
    (
        "p031",
        "uzay",
        "Güneş",
        "Güneş, Güneş Sistemi'nin merkezinde yer alan ve çevresine ısı ile ışık yayan bir "
        "yıldızdır.",
        "Güneş Sistemi'nin merkezindeki yıldız nedir?",
    ),
    (
        "p032",
        "uzay",
        "Ay",
        "Ay, Dünya'nın tek doğal uydusudur ve Dünya'ya dönük yüzü her zaman aynı kalır.",
        "Dünya'nın tek doğal uydusu hangisidir?",
    ),
    (
        "p033",
        "uzay",
        "Mars",
        "Mars, yüzeyindeki demir oksit nedeniyle kızıl renkte görünen ve iki küçük uydusu olan "
        "bir gezegendir.",
        "Kızıl gezegen olarak bilinen gezegen hangisidir?",
    ),
    (
        "p034",
        "uzay",
        "Jüpiter",
        "Jüpiter, Güneş Sistemi'nin en büyük gezegenidir ve Büyük Kırmızı Leke adlı dev bir "
        "fırtınaya sahiptir.",
        "Güneş Sistemi'nin en büyük gezegeni hangisidir?",
    ),
    (
        "p035",
        "uzay",
        "Kara Delikler",
        "Kara delikler, çekim gücü o kadar güçlü olan bölgelerdir ki ışık bile onların "
        "etkisinden kaçamaz.",
        "Işığın bile kaçamadığı gök cismi nedir?",
    ),
    (
        "p036",
        "uzay",
        "Güneş Tutulması",
        "Güneş tutulması, Ay'ın Dünya ile Güneş arasına girmesiyle Güneş'in bir bölümünün "
        "kapanmasıdır.",
        "Güneş tutulması sırasında hangi gök cismi araya girer?",
    ),
    (
        "p037",
        "uzay",
        "Kuyruklu Yıldızlar",
        "Kuyruklu yıldızlar, buz ve tozdan oluşan, Güneş'e yaklaştıkça arkalarında parlak bir "
        "kuyruk bırakan cisimlerdir.",
        "Güneş'e yaklaşınca kuyruk oluşturan gök cisimleri hangileridir?",
    ),
    (
        "p038",
        "uzay",
        "Samanyolu",
        "Samanyolu, içinde Güneş Sistemi'nin bulunduğu sarmal yapılı bir gökadadır.",
        "Güneş Sistemi hangi gökadada bulunur?",
    ),
    (
        "p039",
        "uzay",
        "Yapay Uydular",
        "Yapay uydular, Dünya yörüngesine yerleştirilen ve haberleşme, hava tahmini gibi "
        "işlerde kullanılan araçlardır.",
        "Haberleşme ve hava tahmininde kullanılan yörünge araçlarına ne denir?",
    ),
    (
        "p040",
        "uzay",
        "Venüs",
        "Venüs, kalın karbondioksit atmosferi nedeniyle Güneş Sistemi'nin en sıcak yüzeyine "
        "sahip gezegenidir.",
        "Karbondioksit atmosferi nedeniyle en sıcak yüzeye sahip gezegen hangisidir?",
    ),
    (
        "p041",
        "teknoloji",
        "Transistör",
        "Transistör, elektrik sinyalini yükselten veya anahtarlayan yarı iletken bir elektronik "
        "bileşendir.",
        "Elektrik sinyalini anahtarlayan yarı iletken bileşen nedir?",
    ),
    (
        "p042",
        "teknoloji",
        "İnternet",
        "İnternet, birbirine bağlı bilgisayar ağlarının ortak protokollerle veri paylaşmasını "
        "sağlayan küresel sistemdir.",
        "Bilgisayar ağlarını birbirine bağlayan küresel sistem nedir?",
    ),
    (
        "p043",
        "teknoloji",
        "Açık Kaynak Yazılım",
        "Açık kaynak yazılım, kaynak kodunun herkesin inceleyip değiştirebildiği ve genellikle "
        "özgür bir lisansla dağıtıldığı yazılımdır.",
        "Kaynak kodu herkesin inceleyebildiği yazılım türü nedir?",
    ),
    (
        "p044",
        "teknoloji",
        "İlişkisel Veritabanı",
        "İlişkisel veritabanları, verileri tablolar hâlinde saklar ve tablolar arasında anahtar "
        "ilişkileri kurar.",
        "Verileri tablolar hâlinde saklayan veritabanı türü hangisidir?",
    ),
    (
        "p045",
        "teknoloji",
        "Makine Öğrenmesi",
        "Makine öğrenmesi, bilgisayarların veriden örüntü öğrenerek tahmin yapmasını sağlayan "
        "yöntemler bütünüdür.",
        "Bilgisayarların veriden örüntü öğrenmesini sağlayan alan nedir?",
    ),
    (
        "p046",
        "teknoloji",
        "Şifreleme",
        "Şifreleme, bir mesajı yetkisiz kişilerin okuyamayacağı biçime dönüştürme işlemidir.",
        "Mesajın yetkisiz kişilerce okunmasını engelleyen işlem nedir?",
    ),
    (
        "p047",
        "teknoloji",
        "Bulut Bilişim",
        "Bulut bilişim, hesaplama ve depolama kaynaklarının internet üzerinden kiralanarak "
        "kullanılmasını sağlar.",
        "Hesaplama kaynaklarının internetten kiralanmasına ne denir?",
    ),
    (
        "p048",
        "teknoloji",
        "Yapay Sinir Ağları",
        "Yapay sinir ağları, birbirine bağlı düğümlerden oluşan ve örneklerden öğrenen "
        "hesaplama modelleridir.",
        "Birbirine bağlı düğümlerden oluşan öğrenme modeli nedir?",
    ),
    (
        "p049",
        "teknoloji",
        "Uygulama Programlama Arayüzü",
        "Uygulama programlama arayüzü, iki yazılımın belirli kurallarla birbiriyle veri "
        "alışverişi yapmasını sağlar.",
        "İki yazılımın kurallı veri alışverişini sağlayan arayüz nedir?",
    ),
    (
        "p050",
        "teknoloji",
        "Arama Motorları",
        "Arama motorları, sorguyla ilgili sayfaları bir dizin üzerinden sıralayarak kullanıcıya "
        "sunar.",
        "Sorguya uygun sayfaları dizinden sıralayıp sunan sistem nedir?",
    ),
    (
        "p051",
        "doga",
        "Bal Arıları",
        "Bal arıları, çiçeklerden nektar toplayarak bal üretir ve bu sırada bitkilerin "
        "tozlaşmasına yardımcı olur.",
        "Bal arıları bitkilere hangi yolla katkı sağlar?",
    ),
    (
        "p052",
        "doga",
        "Ahtapot",
        "Ahtapotlar, sekiz kola sahip, mürekkep salgılayabilen ve renk değiştirerek saklanan "
        "deniz canlılarıdır.",
        "Renk değiştirerek saklanan sekiz kollu deniz canlısı hangisidir?",
    ),
    (
        "p053",
        "doga",
        "Penguenler",
        "Penguenler uçamayan ancak yüzmeye uyarlanmış kanatlarıyla soğuk okyanuslarda avlanan "
        "kuşlardır.",
        "Uçamayan ama yüzebilen kutup kuşu hangisidir?",
    ),
    (
        "p054",
        "doga",
        "Ağaç Halkaları",
        "Bir ağacın gövde kesitindeki halkalar, her yıl oluşan büyüme katmanlarını gösterir ve "
        "yaş tahmininde kullanılır.",
        "Ağacın yaşı gövdesindeki hangi yapıdan anlaşılır?",
    ),
    (
        "p055",
        "doga",
        "Bukalemunlar",
        "Bukalemunlar, deri hücrelerindeki renk pigmentlerini değiştirerek çevrelerine uyum "
        "sağlar.",
        "Bukalemunlar çevreye nasıl uyum sağlar?",
    ),
    (
        "p056",
        "doga",
        "Bitki Kökleri",
        "Bitki kökleri, topraktan su ve mineral alır; ayrıca bitkiyi toprağa bağlayarak destek "
        "sağlar.",
        "Bitki köklerinin topraktan aldığı maddeler nelerdir?",
    ),
    (
        "p057",
        "doga",
        "Göçmen Kuşlar",
        "Göçmen kuşlar, mevsim değişimlerinde besin bulmak için uzun mesafeler kat ederek başka "
        "bölgelere gider.",
        "Kuşlar mevsim değişiminde neden göç eder?",
    ),
    (
        "p058",
        "doga",
        "Mercan Resifleri",
        "Mercan resifleri, binlerce küçük polipin salgıladığı kalsiyum karbonat "
        "iskeletlerinden oluşan yapılardır.",
        "Mercan resifleri hangi maddeden oluşan iskeletlerden meydana gelir?",
    ),
    (
        "p059",
        "doga",
        "Karınca Kolonileri",
        "Karınca kolonileri, iş bölümüne dayanan düzenleriyle yiyecek toplar ve yavrularını korur.",
        "Karınca kolonilerinde işler nasıl yürütülür?",
    ),
    (
        "p060",
        "doga",
        "Kaktüsler",
        "Kaktüsler, suyu gövdesinde depolayan ve diken biçimindeki yapraklarıyla su kaybını "
        "azaltan bitkilerdir.",
        "Kaktüsler su kaybını nasıl azaltır?",
    ),
    (
        "p061",
        "kultur",
        "Türk Kahvesi",
        "Türk kahvesi, ince öğütülmüş çekirdeklerin cezvede kaynatılmasıyla hazırlanır ve "
        "telvesiyle servis edilir.",
        "Türk kahvesi nasıl hazırlanır?",
    ),
    (
        "p062",
        "kultur",
        "Nazar Boncuğu",
        "Nazar boncuğu, mavi camdan yapılan ve kötü bakışlardan koruduğuna inanılan geleneksel "
        "bir süstür.",
        "Kötü bakışlardan koruduğuna inanılan mavi cam süs nedir?",
    ),
    (
        "p063",
        "kultur",
        "Karagöz ve Hacivat",
        "Karagöz ve Hacivat, gölge oyunu olarak perde arkasından ışıkla oynatılan geleneksel "
        "Türk tiyatrosudur.",
        "Perde arkasında ışıkla oynatılan geleneksel oyun hangisidir?",
    ),
    (
        "p064",
        "kultur",
        "Ebru Sanatı",
        "Ebru sanatı, su yüzeyine serpilen boyaların kâğıda aktarılmasıyla yapılan bir kâğıt "
        "süsleme sanatıdır.",
        "Su yüzeyindeki boyaların kâğıda aktarılmasıyla yapılan sanat nedir?",
    ),
    (
        "p065",
        "kultur",
        "Sema Töreni",
        "Sema töreni, ney ve bendir eşliğinde dönerek yapılan bir zikir ve gösteri geleneğidir.",
        "Dönerek yapılan zikir ve gösteri geleneğinin adı nedir?",
    ),
    (
        "p066",
        "kultur",
        "Baklava",
        "Baklava, ince açılmış yufkaların arasına fıstık veya ceviz konup şerbetle "
        "tatlandırılmasıyla yapılır.",
        "İnce yufkalar ve şerbetle yapılan tatlı hangisidir?",
    ),
    (
        "p067",
        "kultur",
        "Türk Halıları",
        "Türk halıları, düğüm tekniğiyle dokunan ve motifleriyle yöresel hikâyeler anlatan el "
        "dokuması ürünlerdir.",
        "Motifleriyle yöresel hikâyeler anlatan el dokuması nedir?",
    ),
    (
        "p068",
        "kultur",
        "Türk Çayı",
        "Türk çayı, demlikte demlenen koyu çayın üzerine sıcak su eklenerek ince belli bardakta "
        "servis edilir.",
        "Türk çayı hangi kapta servis edilir?",
    ),
    (
        "p069",
        "kultur",
        "Türk Hamamı",
        "Türk hamamı, sıcak taş yüzey ve buhar odalarıyla yıkanma ve dinlenme geleneğine "
        "dayanan yapıdır.",
        "Sıcak taş ve buhar odalarıyla bilinen yapı hangisidir?",
    ),
    (
        "p070",
        "kultur",
        "Âşıklık Geleneği",
        "Âşıklık geleneği, saz eşliğinde doğaçlama şiir söyleyen halk ozanlarının sürdürdüğü "
        "bir gelenektir.",
        "Saz eşliğinde doğaçlama şiir söyleyenlere ne denir?",
    ),
    (
        "p071",
        "edebiyat",
        "Köy Romanı",
        "Köy romanı, kırsal yaşamı, toprak ilişkilerini ve ağalık düzenini konu edinen bir "
        "roman türüdür.",
        "Kırsal yaşamı ve ağalık düzenini konu edinen roman türü nedir?",
    ),
    (
        "p072",
        "edebiyat",
        "Divan Edebiyatı",
        "Divan edebiyatı, Osmanlı döneminde saray çevresinde gelişen, aruz ölçüsüyle yazılan "
        "klasik şiir geleneğidir.",
        "Aruz ölçüsüyle yazılan Osmanlı klasik şiir geleneği nedir?",
    ),
    (
        "p073",
        "edebiyat",
        "İlahi",
        "İlahi, tasavvufi konuları sade bir Türkçeyle işleyen ve dörtlüklerle söylenen halk "
        "şiiri türüdür.",
        "Tasavvufi konuları sade Türkçeyle işleyen halk şiiri türü nedir?",
    ),
    (
        "p074",
        "edebiyat",
        "Nobel Edebiyat Ödülü",
        "Nobel Edebiyat Ödülü, her yıl edebiyat alanına katkı sağlayan bir yazara verilir.",
        "Nobel Edebiyat Ödülü hangi alanda verilir?",
    ),
    (
        "p075",
        "edebiyat",
        "Masal Kalıpları",
        "Türk masallarında sık görülen “bir varmış bir yokmuş” kalıbı, anlatının olağanüstü "
        "dünyaya açıldığını bildirir.",
        "Masalların başında kullanılan kalıplaşmış ifade nedir?",
    ),
    (
        "p076",
        "edebiyat",
        "Mani",
        "Mani, dört dizeden oluşan ve genellikle aşk ile ayrılık temalarını işleyen kısa halk "
        "şiiri türüdür.",
        "Dört dizeden oluşan kısa halk şiiri türü nedir?",
    ),
    (
        "p077",
        "edebiyat",
        "Türk Romanı",
        "Tanzimat döneminde çeviriyle başlayan Türk romanı, zamanla toplumsal sorunları işleyen "
        "özgün eserler üretti.",
        "Türk romanı hangi dönemde çevirilerle başladı?",
    ),
    (
        "p078",
        "edebiyat",
        "Oğuz Destanları",
        "Oğuz boylarının kahramanlıklarını anlatan destansı halk anlatıları, sözlü gelenek "
        "içinde kuşaktan kuşağa aktarılmıştır.",
        "Oğuz boylarının kahramanlıklarını anlatan anlatılar hangi gelenekle aktarılmıştır?",
    ),
    (
        "p079",
        "edebiyat",
        "Servet-i Fünun Şiiri",
        "Servet-i Fünun döneminde şiir, bireysel duygularla toplumsal eleştiriyi "
        "birleştiren bir anlayışa yöneldi.",
        "Servet-i Fünun döneminde bireysel duygularla toplumsal eleştiriyi birleştiren tür nedir?",
    ),
    (
        "p080",
        "edebiyat",
        "Atasözleri",
        "Atasözleri, uzun deneyimlerden süzülen öğütleri kısa ve kalıplaşmış cümlelerle aktaran "
        "halk sözleridir.",
        "Deneyimlerden süzülen öğütleri aktaran kalıplaşmış sözler nedir?",
    ),
    (
        "p081",
        "spor",
        "Futbol",
        "Futbolda bir takım sahaya on bir oyuncuyla çıkar ve topu rakip kaleye sokarak gol "
        "atmaya çalışır.",
        "Futbolda bir takım sahaya kaç oyuncuyla çıkar?",
    ),
    (
        "p082",
        "spor",
        "Basketbol",
        "Basketbolda sayı, topun rakip potadan geçirilmesiyle kazanılır; üç sayı çizgisinin "
        "gerisinden atılan isabetli şut üç sayı değerindedir.",
        "Basketbolda üç sayı çizgisinin gerisinden atılan şut kaç sayıdır?",
    ),
    (
        "p083",
        "spor",
        "Voleybol",
        "Voleybolda amaç, topu file üzerinden rakip sahaya düşürmek ve rakibin oyunu "
        "sürdürmesini engellemektir; bir takım topa en fazla üç kez dokunabilir.",
        "Voleybolda takımlar topa en fazla kaç dokunuş yapabilir?",
    ),
    (
        "p084",
        "spor",
        "Olimpiyat Oyunları",
        "Modern Olimpiyat Oyunları dört yılda bir düzenlenir; yaz ve kış oyunları ayrı "
        "takvimlerde yapılır.",
        "Modern Olimpiyat Oyunları kaç yılda bir düzenlenir?",
    ),
    (
        "p085",
        "spor",
        "Yağlı Güreş",
        "Yağlı güreş, er meydanında yapılan geleneksel bir Türk sporudur ve Kırkpınar'da "
        "sürdürülen turnuvayla tanınır.",
        "Kırkpınar'da düzenlenen geleneksel güreş türü nedir?",
    ),
    (
        "p086",
        "spor",
        "Maraton",
        "Maraton koşusu 42,195 kilometrelik bir parkurda yapılan uzun mesafe koşusudur.",
        "Maraton koşusunun mesafesi ne kadardır?",
    ),
    (
        "p087",
        "spor",
        "Serbest Yüzme",
        "Serbest yüzmede sporcular kulaç hareketiyle ilerler ve her kulaçta yüzlerini yana "
        "çevirerek nefes alır.",
        "Serbest yüzmede sporcular nasıl nefes alır?",
    ),
    (
        "p088",
        "spor",
        "Tenis",
        "Teniste oyuncular topu raketle karşı sahadaki sınırlar içine atarak puan kazanmaya "
        "çalışır.",
        "Teniste puan kazanmak için top nereye atılır?",
    ),
    (
        "p089",
        "spor",
        "Formula 1",
        "Formula 1 yarışları, özel olarak tasarlanmış tek kişilik araçlarla kapalı pistlerde "
        "yapılan otomobil yarışlarıdır.",
        "Tek kişilik özel araçlarla yapılan otomobil yarışı hangisidir?",
    ),
    (
        "p090",
        "spor",
        "Bisiklet Turları",
        "Tur bisikleti yarışlarında sporcular etap etap ilerler ve genel sıralama toplam "
        "süreye göre belirlenir.",
        "Bisiklet turlarında sporcular hangi ölçüte göre sıralanır?",
    ),
    (
        "p091",
        "dil",
        "Büyük Ünlü Uyumu",
        "Türkçede büyük ünlü uyumu, bir sözcükteki ünlülerin kalınlık-incelik bakımından uyumlu "
        "olmasını gerektirir.",
        "Türkçede kalınlık-incelik uyumunu düzenleyen kural nedir?",
    ),
    (
        "p092",
        "dil",
        "Eklemeli Dil",
        "Türkçe eklemeli bir dildir; sözcükler kök ve eklerin birleşmesiyle yeni anlamlar kazanır.",
        "Türkçe yapı bakımından hangi dil tipine girer?",
    ),
    (
        "p093",
        "dil",
        "Fiilde Çatı",
        "Fiilde çatı, öznenin ve nesnenin fiille ilişkisini belirler; etken, edilgen ve dönüşlü "
        "gibi türleri vardır.",
        "Fiilde özne ve nesnenin fiille ilişkisini belirleyen dil bilgisi kategorisi nedir?",
    ),
    (
        "p094",
        "dil",
        "Sözcük Türleri",
        "Türkçede sözcükler isim, sıfat, zamir, zarf, edat, bağlaç ve ünlem gibi türlere ayrılır.",
        "Türkçede sözcükler hangi türlere ayrılır?",
    ),
    (
        "p095",
        "dil",
        "Virgül",
        "Virgül, eş görevli sözcükleri ayırmak ve anlam karışıklığını önlemek için kullanılan "
        "bir noktalama işaretidir.",
        "Eş görevli sözcükleri ayırmak için hangi noktalama işareti kullanılır?",
    ),
    (
        "p096",
        "dil",
        "“de” Bağlacı",
        "Türkçede “de” bağlacı ayrı yazılır; bulunma durumu eki olan “-de” ise sözcüğe bitişik "
        "yazılır.",
        "“de” bağlacı nasıl yazılır?",
    ),
    (
        "p097",
        "dil",
        "Deyimler",
        "Deyimler, en az iki sözcükten oluşan ve gerçek anlamından uzaklaşmış kalıplaşmış "
        "ifadelerdir.",
        "Gerçek anlamından uzaklaşmış kalıplaşmış ifadeler nedir?",
    ),
    (
        "p098",
        "dil",
        "Ünsüz Yumuşaması",
        "Ünsüz yumuşaması, sonu p, ç, t, k ile biten sözcüklere ünlüyle başlayan ek geldiğinde "
        "bu seslerin yumuşamasıdır.",
        "Sert ünsüzlerin yumuşamasına ne ad verilir?",
    ),
    (
        "p099",
        "dil",
        "Hece",
        "Türkçede her hecede bir ünlü bulunur; bu nedenle sözcükteki hece sayısı ünlü sayısına "
        "eşittir.",
        "Bir sözcükteki hece sayısı neye eşittir?",
    ),
    (
        "p100",
        "dil",
        "Kurallı Cümle",
        "Türkçede kurallı cümlede yüklem sonda yer alır; devrik cümlede ise yüklem başka bir "
        "konuma taşınır.",
        "Türkçede kurallı cümlede yüklem nerede bulunur?",
    ),
]
