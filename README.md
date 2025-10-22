# Akıllı Öğrenci Asistanı 🎓

**Kapsamlı bir RAG (Retrieval-Augmented Generation) altyapısına sahip, üretime hazır Yapay Zekâ Destekli Öğrenci Asistanı.
Google Gemini 2.5 Flash modeliyle çalışarak; kişisel öğrenme analitiği, hedef takibi, PDF ve web içerik analizi ile öğrencilerin verimli şekilde çalışmasına yardımcı olur.
Ayrıca, Hugging Face üzerinde barındırılan özel embedding modeli sayesinde kendi dokümanlarınız üzerinden akıllı yanıtlar üretir.**


#### Not: Lütfen uygulamayı başlattığınızda Hugging Face'den embedding modeli indirilirken ve pdf işlenirken başka bir işlem yapmayın. (Tavsiye edilir.)
 **Rag Chatbot menüsünde uygun soruları sormak için rag_pdfs klasöründeki PDF dosyasını inceleyiniz.**
###### Örneğin; Üniversite öğrencilerinin zaman yönetimi davranışları ve bu davranışların akademik başarı ile ilişkisi nedir ? 
 



## Temel Özellikler

###### Akıllı Sohbet: *Google Gemini 2.5 Flash altyapısını kullanan, geçmiş konuşmaları hatırlayabilen bir chatbot.*

###### Öğrenme Analitiği: *Çalışma sürelerini ve konuları kaydederek yapay zeka destekli kişisel çalışma önerileri sunar.*

###### Görsel Pano: *Matplotlib kullanarak son 7 günlük çalışma performansını gösteren interaktif grafikler.*

###### Hedef Takip Sistemi: *Öğrencilerin hedefler belirlemesini, ilerlemelerini kaydetmesini ve tamamlanan hedefleri görmesini sağlar.*

###### PDF İşleme: *Yüklenen herhangi bir PDF dosyasını özetleyebilir veya dosya içeriği hakkında soruları yanıtlayabilir (Gemini multimodal yeteneği ile).*

###### Web Analizi: *Verilen bir URL'deki web sitesi içeriğini analiz edebilir, özetleyebilir ve içerik hakkında soruları yanıtlayabilir.*

###### RAG Chatbot (Yerel Bilgi Bankası):
###### Belirli bir klasördeki (`rag\_pdfs`) PDF'leri otomatik olarak indeksler.
###### Bilgiyi `FAISS` vektör veritabanında saklar.
###### `LangChain` ve `HuggingFace` (Fuurkan/chatbot-instructor-model) embedding modeli kullanarak bu belgelere özel soruları yanıtlar.

###### Kişiselleştirilmiş Arayüz:Akıcı animasyonlara sahip özel bir giriş (login) ve çıkış (logout) ekranı.



## 🚀 Teknoloji Yığını

*Arayüz (Frontend):Streamlit*

*Yapay Zeka (AI):Google Gemini 2.5 Flash, LangChain*

*RAG: FAISS (Vektör Veritabanı), HuggingFace Instruct Embeddings (`Fuurkan/chatbot-instructor-model`)*

*Veri İşleme: PyPDF2 (PDF), BeautifulSoup (Web)*

*Görselleştirme: Matplotlib*



## 🤖 RAG Modülü: Detaylı Çalışma Prensibi ---- Mentor İsteği  

Projedeki RAG (Retrieval-Augmented Generation) Chatbot, kullanıcının sorduğu soruları, `rag\_pdfs` klasöründeki belgelere dayanarak cevaplamak için tasarlanmıştır. Bu süreç, "indeksleme" ve "sorgulama" olmak üzere iki ana akışa ayrılır.

#### 1. Veri İşleme ve İndeksleme (Uygulama Başlangıcı)

Bu akış, RAG sekmesi ilk açıldığında `initialize\_vector\_store` fonksiyonu ile tetiklenir ve `rag\_pdfs` klasöründeki verileri işler:

1. Veri Okuma (Data Ingestion): `rag\_pdfs` klasöründeki tüm `.pdf` dosyaları `PyPDF2` kullanılarak taranır. Her bir PDF'in metin içeriği sayfa sayfa okunur ve tek bir büyük metin bloğu olarak birleştirilir.

2. Parçalama (Chunking): Bu büyük metin bloğu, `LangChain`'in `RecursiveCharacterTextSplitter`'ı ile daha küçük, yönetilebilir parçalara (chunk) ayrılır (Örn: 1000 karakterlik, 200 karakter örtüşmeli parçalar).

3. Vektör Dönüşümü (Embedding): Bu aşamada `HuggingFaceInstructEmbeddings(model\_name="Fuurkan/chatbot-instructor-model")` fonksiyonu çağrılır.
Eğer model (`Fuurkan/chatbot-instructor-model`) bilgisayarda mevcut değilse, `langchain` kütüphanesi modeli `Hugging Face Hub'dan` otomatik olarak indirir. (Bu ilk çalıştırmada internet bağlantısı gerektirir ve biraz zaman alabilir).

4. İndirilen model, her bir metin parçasını (chunk) sayısal bir vektöre dönüştürür.

5. Vektör Depolama (Vector Store): Oluşturulan tüm bu vektörler, `FAISS` adı verilen yüksek performanslı bir vektör veritabanına yüklenir. Bu veritabanı, oturum süresince (`st.session\_state`) hafızada tutulur.


### 2. Sorgu Akışı (Kullanıcı Entegrasyonu)

Kullanıcı RAG sekmesinden bir soru sorduğunda, aşağıdaki akış gerçekleşir:

1. Kullanıcı Girdisi: Kullanıcı metin kutusuna bir soru yazar.

2. Sorgu Vektörleştirme: Kullanıcının sorusu, 3. adımdaki aynı embedding modeli kullanılarak anında bir sorgu vektörüne dönüştürülür.

3. Benzerlik Araması (Retrieval): `FAISS` veritabanı, bu sorgu vektörüne anlamsal olarak en çok benzeyen (en ilgili) metin parçalarını bulur.

4. Bağlam (Context) Oluşturma: Bulunan bu ilgili metin parçaları, bir "bağlam" (context) olarak bir araya getirilir.

5. Cevap Üretme (Generation): Bu bağlam ve kullanıcının orijinal sorusu, `LangChain`'in `load\_qa\_chain`'i gönderir.

6. Sonuç: Verilen bağlamı (PDF'lerden gelen bilgiyi) kullanarak kullanıcının sorusunu yanıtlar ve bu cevap ekranda gösterilir.



## ⚙️ Kurulum ve Çalıştırma

### 1. Gereksinimler

 *Python 3.10+*
 
*Google Gemini 2.5 Flash erişimi olan bir Google API Anahtarı.*

*Aktif İnternet Bağlantısı: RAG modülünün ilk çalıştırılmasında Hugging Face'den embedding modelini (yaklaşık 1.3 GB) otomatik olarak indirmek için gereklidir.*



### 2. Projeyi Klonlama

```bash

git clone https://github.com/furkantutanc/akilli-ogrenci-asistani
cd akilli-ogrenci-asistani
```


### 3 Gerekli Paketleri Yükleme

Proje için gerekli tüm bağımlılıkları requirements.txt dosyasından yükleyin:
```
pip install -r requirements.txt
```


**4 RAG Veri Hazırlığı (Önemli)**

*RAG Chatbot modülü, rag\_pdfs klasöründeki PDF dosyalarından beslenir. Projeyi klonladığınızda bu klasör ve içindeki örnek PDF'ler otomatik olarak gelecektir.*





5\. Uygulamayı Başlatma
```
streamlit run app.py
```


Giriş ekranında sizden Adınız ve Google API Anahtarınızı girmeniz istenecektir.**

### Deploy edilen web arayüzünün linki: https://akilli-ogrenci-asistani-ft.streamlit.app/

