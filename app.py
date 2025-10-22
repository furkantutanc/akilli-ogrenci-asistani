import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
import PyPDF2
import requests
import json
import datetime
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import os
import random
import time
import glob
from typing import Dict, List, Any

# --- RAG İÇİN İMPORTLAR ---
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI


# ==================== KİŞİSEL ÖĞRENME ANALİTİĞİ ====================
class OgrenmeAnalitigi:
    def __init__(self):
        self.calisma_kayitlari = {}  # {kullanici_adi: [{ders, sure, tarih, konular}, ...]}
        self.ders_istatistikleri = {}
    
    def calisma_kaydet(self, kullanici_adi: str, ders: str, sure: float, konular: List[str]):
        """Çalışma kaydı ekle"""
        if kullanici_adi not in self.calisma_kayitlari:
            self.calisma_kayitlari[kullanici_adi] = []
        
        kayit = {
            "ders": ders,
            "sure": sure,
            "konular": konular,
            "tarih": datetime.datetime.now(),
            "gun": datetime.datetime.now().date()
        }
        self.calisma_kayitlari[kullanici_adi].append(kayit)
    
    def son_calisma_zamani(self, kullanici_adi: str, ders: str) -> int:
        """Belirli bir derste son çalışmadan bu yana geçen gün sayısı"""
        if kullanici_adi not in self.calisma_kayitlari:
            return -1
        
        ders_kayitlari = [k for k in self.calisma_kayitlari[kullanici_adi] if k["ders"].lower() == ders.lower()]
        if not ders_kayitlari:
            return -1
        
        son_kayit = max(ders_kayitlari, key=lambda x: x["tarih"])
        gecen_gun = (datetime.datetime.now().date() - son_kayit["gun"]).days
        return gecen_gun
    
    def ders_analizi(self, kullanici_adi: str, ders: str) -> Dict:
        """Belirli bir ders için detaylı analiz"""
        if kullanici_adi not in self.calisma_kayitlari:
            return {"hata": "Veri bulunamadı"}
        
        ders_kayitlari = [k for k in self.calisma_kayitlari[kullanici_adi] if k["ders"].lower() == ders.lower()]
        
        if not ders_kayitlari:
            return {
                "ders": ders,
                "toplam_sure": 0,
                "calisma_gun_sayisi": 0,
                "son_calisma": -1,
                "ortalama_sure": 0,
                "toplam_konu": 0,
                "durum": "yeni"
            }
        
        toplam_sure = sum([k["sure"] for k in ders_kayitlari])
        calisma_gunleri = len(set([k["gun"] for k in ders_kayitlari]))
        son_calisma = self.son_calisma_zamani(kullanici_adi, ders)
        ortalama_sure = toplam_sure / calisma_gunleri if calisma_gunleri > 0 else 0
        toplam_konu = sum([len(k["konular"]) for k in ders_kayitlari])
        
        # Son 7 günde çalışma kontrolü
        son_7_gun = [k for k in ders_kayitlari if (datetime.datetime.now().date() - k["gun"]).days <= 7]
        
        return {
            "ders": ders,
            "toplam_sure": toplam_sure,
            "calisma_gun_sayisi": calisma_gunleri,
            "son_calisma": son_calisma,
            "ortalama_sure": ortalama_sure,
            "toplam_konu": toplam_konu,
            "son_7_gun_calisma": len(son_7_gun),
            "durum": "aktif" if son_calisma <= 2 else "durgun"
        }
    
    def genel_analiz(self, kullanici_adi: str) -> Dict:
        """Tüm dersler için genel analiz"""
        if kullanici_adi not in self.calisma_kayitlari:
            return {}
        
        dersler = list(set([k["ders"] for k in self.calisma_kayitlari[kullanici_adi]]))
        analiz_sonuclari = {}
        
        for ders in dersler:
            analiz_sonuclari[ders] = self.ders_analizi(kullanici_adi, ders)
        
        return analiz_sonuclari
    
    def ai_onerileri_olustur(self, kullanici_adi: str, api_anahtari: str = "") -> str:
        """AI tabanlı öneriler oluştur"""
        if not api_anahtari:
            return "API anahtarı gerekli."
        
        genel_analiz = self.genel_analiz(kullanici_adi)
        
        if not genel_analiz:
            return "Henüz çalışma kaydı yok. Çalışmaya başladığında analiz yapabileceğim!"
        
        # Analiz verilerini metin formatına çevir
        analiz_metni = "**ÇALIŞMA ANALİZİ:**\n\n"
        for ders, veri in genel_analiz.items():
            analiz_metni += f"• **{ders}:**\n"
            analiz_metni += f"  - Toplam çalışma: {veri['toplam_sure']:.1f} saat\n"
            analiz_metni += f"  - Çalışılan gün: {veri['calisma_gun_sayisi']} gün\n"
            analiz_metni += f"  - Son çalışma: {veri['son_calisma']} gün önce\n"
            analiz_metni += f"  - Durum: {veri['durum']}\n\n"
        
        try:
            genai.configure(api_key=api_anahtari)
            model = genai.GenerativeModel('gemini-2.5-pro')
            
            prompt = f"""
            Sen bir kişisel öğrenme koçusun. Aşağıdaki öğrenci çalışma verilerini analiz et ve 
            kişiselleştirilmiş ve motive edici önerilerde bulun. Aynı zamanda eğlenceli ol. 
            
            {analiz_metni}
            
            Lütfen şu formatta Türkçe geri bildirim ver:
            
            **📊 GENEL DEĞERLENDİRME:**
            (Genel çalışma alışkanlıkları hakkında kısa bir değerlendirme)
            
            **⚠️ DİKKAT GEREKTİREN DERSLER:**
            (Uzun süredir çalışılmayan veya ihmal edilen dersler)
            
            **✨ GÜÇLÜ YÖNLER:**
            (Düzenli çalışılan ve ilerleme kaydedilen dersler)
            
            **🎯 KİŞİSELLEŞTİRİLMİŞ ÖNERİLER:**
            (Her ders için spesifik öneriler ve çalışma stratejileri)
            
            **💪 MOTİVASYON MESAJI:**
            (Öğrenciyi motive edecek pozitif bir mesaj)
            
            Önerilerini eğlenceli, destekleyici ve uygulanabilir yap. Emoji kullanarak daha samimi ol.
            """
            
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"AI analizi oluşturulamadı: {str(e)}"
    
    def hizli_oneriler(self, kullanici_adi: str) -> List[str]:
        """Hızlı basit öneriler (AI olmadan)"""
        oneriler = []
        genel_analiz = self.genel_analiz(kullanici_adi)
        
        if not genel_analiz:
            return ["📝 Çalışma kaydetmeye başla ve gelişimini takip et!"]
        
        for ders, veri in genel_analiz.items():
            if veri["son_calisma"] > 3:
                oneriler.append(f"⚠️ **{ders}** dersinde {veri['son_calisma']} gündür ilerleme yok!")
            elif veri["son_7_gun_calisma"] >= 5:
                oneriler.append(f"🔥 **{ders}** dersinde harika ilerliyorsun! Son 7 günde {veri['son_7_gun_calisma']} gün çalıştın!")
            elif veri["durum"] == "aktif" and veri["toplam_sure"] > 5:
                oneriler.append(f"💪 **{ders}** dersinde düzenli çalışıyorsun, böyle devam et!")
        
        if not oneriler:
            oneriler.append("👍 Düzenli çalışmalarına devam et!")
        
        return oneriler[:5]  # En fazla 5 öneri

    def gorsel_pano_olustur(self, kullanici_adi: str):
        """Son 7 günlük veriyi analiz eder ve görsel pano için grafikler oluşturur."""
        
        if kullanici_adi not in self.calisma_kayitlari or not self.calisma_kayitlari[kullanici_adi]:
            return None, None, "📊 Pano oluşturmak için henüz yeterli veri yok. Lütfen 'Öğrenme Analitiği' sekmesinden çalışma kaydı ekleyin."

        # Son 7 günün verilerini filtrele
        bugun = datetime.datetime.now().date()
        bir_hafta_once = bugun - datetime.timedelta(days=7)
        haftalik_kayitlar = [k for k in self.calisma_kayitlari[kullanici_adi] if k['gun'] > bir_hafta_once]

        if not haftalik_kayitlar:
            return None, None, "📊 Bu hafta için henüz bir çalışma kaydı bulunmuyor. Çalışmaya devam!"

        # Veriyi işle: Derslere göre toplam süre
        ders_sureleri = {}
        for kayit in haftalik_kayitlar:
            ders = kayit['ders']
            sure = kayit['sure']
            ders_sureleri[ders] = ders_sureleri.get(ders, 0) + sure
            
        if not ders_sureleri:
            return None, None, "Veri işlenirken bir sorun oluştu."

        # Özet metni oluştur
        toplam_haftalik_sure = sum(ders_sureleri.values())
        en_cok_calisilan_ders = max(ders_sureleri, key=ders_sureleri.get)
        ozet_metni = f"Bu hafta toplam **{toplam_haftalik_sure:.1f} saat** çalıştın. En çok **{en_cok_calisilan_ders}** dersine odaklandın. Harika gidiyorsun! 🚀"

        # Stil ayarları (Koyu tema için)
        plt.style.use('dark_background')
        
        # --- 1. Grafik: Ders Süreleri Bar Grafiği ---
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        dersler = list(ders_sureleri.keys())
        sureler = list(ders_sureleri.values())
        
        bars = ax1.bar(dersler, sureler, color='#667eea')
        ax1.set_title('Bu Haftanın Ders Çalışma Süreleri', fontsize=16, color='white', pad=20)
        ax1.set_ylabel('Toplam Saat', fontsize=12, color='white')
        ax1.tick_params(axis='x', colors='white', rotation=45)
        ax1.tick_params(axis='y', colors='white')
        ax1.grid(axis='y', linestyle='--', alpha=0.2)
        
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}s', va='bottom', ha='center', color='white')

        fig1.patch.set_facecolor('#0a0a0a')
        ax1.set_facecolor((1.0, 1.0, 1.0, 0.03))


        # --- 2. Grafik: Günlük Çalışma Dağılımı ---
        gunluk_sureler = {}
        for i in range(7):
            tarih = bugun - datetime.timedelta(days=i)
            gunluk_sureler[tarih.strftime('%a')] = 0
        
        for kayit in haftalik_kayitlar:
            gun_adi = kayit['gun'].strftime('%a')
            gunluk_sureler[gun_adi] += kayit['sure']
            
        gunler = list(gunluk_sureler.keys())[::-1]
        gun_sureleri = list(gunluk_sureler.values())[::-1]

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(gunler, gun_sureleri, marker='o', linestyle='-', color='#764ba2', linewidth=3, markersize=8)
        ax2.set_title('Son 7 Günlük Çalışma Aktivitesi', fontsize=16, color='white', pad=20)
        ax2.set_ylabel('Toplam Saat', fontsize=12, color='white')
        ax2.fill_between(gunler, gun_sureleri, color='#764ba2', alpha=0.2)
        
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')
        ax2.grid(True, linestyle='--', alpha=0.2)
        
        fig2.patch.set_facecolor('#0a0a0a')
        ax2.set_facecolor((1.0, 1.0, 1.0, 0.03))
        
        return fig1, fig2, ozet_metni

# ==================== HEDEF TAKİP SİSTEMİ ====================
class HedefTakipSistemi:
    def __init__(self):
        self.hedefler = {}
        # self.tamamlanan_hedefler = {}
    
    def hedef_ekle(self, kullanici_adi: str, hedef_metni: str, bitis_tarihi: datetime.date, kategori: str = "Genel"):
        if kullanici_adi not in self.hedefler:
            self.hedefler[kullanici_adi] = []
        
        
        next_id = max([h['id'] for h in self.hedefler[kullanici_adi]]) + 1 if self.hedefler[kullanici_adi] else 1
        
        hedef = {
            "id": next_id,
            "metin": hedef_metni,
            "kategori": kategori,
            "olusturma_tarihi": datetime.datetime.now(),
            "bitis_tarihi": bitis_tarihi,
            "tamamlandi": False,
            "ilerleme": 0
        }
        self.hedefler[kullanici_adi].append(hedef)
        return hedef
    
    def hedef_tamamla(self, kullanici_adi: str, hedef_id: int):
        if kullanici_adi in self.hedefler:
            for hedef in self.hedefler[kullanici_adi]:
                if hedef["id"] == hedef_id:
                    hedef["tamamlandi"] = True
                    hedef["ilerleme"] = 100
                    hedef["tamamlanma_tarihi"] = datetime.datetime.now()
                    
                    return True
        return False
    
    def hedef_guncelle(self, kullanici_adi: str, hedef_id: int, ilerleme: int):
        if kullanici_adi in self.hedefler:
            for hedef in self.hedefler[kullanici_adi]:
                if hedef["id"] == hedef_id:
                    hedef["ilerleme"] = min(ilerleme, 100)
                    if ilerleme >= 100:
                        hedef["tamamlandi"] = True     # direkt durumu güncelliyoruz
                        if "tamamlanma_tarihi" not in hedef:
                            hedef["tamamlanma_tarihi"] = datetime.datetime.now()
                    else:
                        hedef["tamamlandi"] = False          # eğer ilerleme %100'den düşerse tekrar aktif hale getirmek için 
                    return True
        return False
    
    def hedef_sil(self, kullanici_adi: str, hedef_id: int):
        if kullanici_adi in self.hedefler:
            self.hedefler[kullanici_adi] = [h for h in self.hedefler[kullanici_adi] if h["id"] != hedef_id]
            return True
        return False
    
    def aktif_hedefler(self, kullanici_adi: str) -> List[Dict]:
        if kullanici_adi not in self.hedefler:
            return []
        return [h for h in self.hedefler[kullanici_adi] if not h["tamamlandi"]]
    
    def tamamlanan_hedefler_listesi(self, kullanici_adi: str) -> List[Dict]:
        if kullanici_adi not in self.hedefler:
            return []
        return [h for h in self.hedefler[kullanici_adi] if h["tamamlandi"]]
    
    def hatirlatma_kontrol(self, kullanici_adi: str) -> List[str]:
        hatirlatmalar = []
        bugun = datetime.datetime.now().date()
        
        for hedef in self.aktif_hedefler(kullanici_adi):
            bitis = hedef["bitis_tarihi"]
            kalan_gun = (bitis - bugun).days
            
            if kalan_gun < 0:
                hatirlatmalar.append(f"🔴 **GECIKMIŞ:** {hedef['metin']} ({abs(kalan_gun)} gün geçti)")
            elif kalan_gun == 0:
                hatirlatmalar.append(f"🟡 **BUGÜN BİTİYOR:** {hedef['metin']}")
            elif kalan_gun <= 2:
                hatirlatmalar.append(f"🟠 **{kalan_gun} GÜN KALDI:** {hedef['metin']}")
        
        return hatirlatmalar[:3]      # En fazla 3 hatırlatma göstercek şekilde yaptım
    
    def hedef_istatistikleri(self, kullanici_adi: str) -> Dict:
        if kullanici_adi not in self.hedefler or not self.hedefler[kullanici_adi]:
            return {"aktif_hedef": 0, "tamamlanan_hedef": 0, "toplam_hedef": 0, "tamamlanma_orani": 0}

        aktif = len(self.aktif_hedefler(kullanici_adi))
        tamamlanan = len(self.tamamlanan_hedefler_listesi(kullanici_adi))
        toplam = aktif + tamamlanan
        
        tamamlanma_orani = (tamamlanan / toplam * 100) if toplam > 0 else 0
        
        return {
            "aktif_hedef": aktif,
            "tamamlanan_hedef": tamamlanan,
            "toplam_hedef": toplam,
            "tamamlanma_orani": tamamlanma_orani
        }
# ==================== MOTİVASYON SİSTEMİ ====================
class MotivasyonSistemi:
    def __init__(self):
        self.motivasyon_mesajlari = [
            "Ya yaparsın, ya da bahane bulup kenara çekilirsin.", "Bahaneler değil, sonuçlar konuşur.! ", "Kimse sana inanmak zorunda değil, kanıtla!",
            "Kazananlar bahaneyi değil, yolu bulur.", "Bugün yorgunsan, yarın gurur duyarsın."
        ]
        # self.ilerleme_verisi = {} 

    def rastgele_motivasyon(self):
        return random.choice(self.motivasyon_mesajlari)


# ==================== PDF İŞLEME MODÜLÜ ====================
class PDFIsleyici:
    def __init__(self):
        self.pdf_veritabani = {}
        self.mevcut_pdf_bytes = None

    def pdf_yukle(self, pdf_dosyasi):
        """PDF dosyasını byte olarak sakla"""
        try:
            self.mevcut_pdf_bytes = pdf_dosyasi.read()
            pdf_dosyasi.seek(0)  # Dosya pointer'ını başa alma
            return True
        except Exception as e:
            st.error(f"PDF yükleme hatası: {str(e)}")
            return False

    def pdf_ozetle(self, pdf_dosyasi, api_anahtari: str = "") -> str:
        if not api_anahtari:
            return "Lütfen önce API anahtarınızı giriniz."
        try:
            genai.configure(api_key=api_anahtari)
            model = genai.GenerativeModel('gemini-2.5-pro')

            
            
            pdf_bytes = pdf_dosyasi.read()
            pdf_dosyasi.seek(0)
            
            pdf_part = {
                'mime_type': 'application/pdf',
                'data': pdf_bytes
            }
            
            prompt = """
            Bu PDF dosyasını analiz et ve aşağıdaki formatta Türkçe özet çıkar:
            
            **📋 ÖZET:**
            • Ana konu ve içerik hakkında işe yarar özet (5-7 madde)
            
            **🔑 ANAHTAR KELİMELER:**
            • En önemli 5-7 anahtar kelime
            
            **💡 ÖNEMLİ NOKTALAR:**
            • Dikkat çekilmesi gereken önemli bilgiler
            """
            
            
            response = model.generate_content([prompt, pdf_part])
            
            
            return response.text
            
        except Exception as e:
            return f"Özetleme hatası: {str(e)}\n\n**İpucu:** API anahtarınızın doğru olduğundan ve 'Generative Language API' izninin aktif olduğundan emin olun."

    def pdf_soru_cevapla(self, pdf_dosyasi, soru: str, api_anahtari: str = "") -> str:
        if not api_anahtari:
            return "Lütfen önce API anahtarınızı giriniz."
        try:
            genai.configure(api_key=api_anahtari)
            model = genai.GenerativeModel('gemini-2.5-pro')

            
            pdf_bytes = pdf_dosyasi.read()
            pdf_dosyasi.seek(0)
            
            pdf_part = {
                'mime_type': 'application/pdf',
                'data': pdf_bytes
            }
            
            prompt = f"""
            Bu PDF dosyasını analiz et ve aşağıdaki soruyu Türkçe olarak yanıtla:
            
            SORU: {soru}
            
            Lütfen açıklayıcı bir cevap ver. Gerekirse PDF'teki ilgili bölümlere atıfta bulun.
            """
            
            # Modeli, prompt ve dosya bölümü ile birlikte çağırıyoruz.
            response = model.generate_content([prompt, pdf_part])
            
            
            return response.text
            
        except Exception as e:
            return f"Soru cevaplama hatası: {str(e)}"
# ==================== WEB ANALİZ MODÜLÜ ====================
class WebAnaliz:
    def __init__(self):
        self.web_icerikleri = {}
        self.mevcut_url = None
        self.mevcut_icerik = None

    def _clean_html_content(self, soup: BeautifulSoup) -> str:
        """HTML'den temiz metin içeriği çıkarmak için yardımcı fonksiyon."""
        
        # 1. Öncelikli etiketleri aramak için
        main_content = soup.find('article')
        if not main_content:
            main_content = soup.find('main')
        
        if main_content:
            # Bu etiketlerin içindeki olası (navigasyon, reklam vb.) temizle TEMİZ VERİ İÇİN
            for tag in main_content(['nav', 'aside', 'footer', 'header', 'script', 'style', 'button', 'a']):
                tag.decompose()
            return main_content.get_text(separator='\n', strip=True)

        # 2. Yaygın ID/Class'ları aramak (Bloglar, haber siteleri için)
        common_ids = ['content', 'main-content', 'main', 'post-body', 'article-body']
        common_classes = ['content', 'main-content', 'post-content', 'entry-content', 'article-content']
        
        for id_name in common_ids:
            content_block = soup.find(id=id_name)
            if content_block:
                return content_block.get_text(separator='\n', strip=True)
        
        for class_name in common_classes:
            content_block = soup.find(class_=class_name)
            if content_block:
                return content_block.get_text(separator='\n', strip=True)

        # 3. Fallback: Body'den gürültüyü temizle 
        body = soup.find('body')
        if not body:
            return soup.get_text(separator='\n', strip=True) # Sadece metin varsa

        # Gürültü etiketlerini kaldır
        for tag in body(['nav', 'aside', 'footer', 'header', 'script', 'style', 'a', 'button']):
            tag.decompose()
            
        text = body.get_text(separator='\n', strip=True)
        
        # Çok fazla boş satırı temizle
        lines = [line for line in text.split('\n') if line.strip()]
        return "\n".join(lines)

    def web_sitesi_oku(self, url: str, api_anahtari: str = "") -> Dict[str, Any]:
        """Web sitesini oku ve Gemini ile analiz et (GELİŞTİRİLMİŞ TEMİZLEME)"""
        if not api_anahtari:
            return {"hata": "API anahtarı gerekli"}
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            baslik = soup.find('title').text if soup.find('title') else "Başlık Yok"
            
            
            # icerik = "\n".join([p.text for p in soup.find_all('p')]) # ESKİ KOD
            icerik = self._clean_html_content(soup)
            
            
            if not icerik.strip():
                 return {"hata": "Bu web sitesinden metin içeriği çekilemedi. Site, dinamik (JavaScript) içerik kullanıyor olabilir veya erişim engellidir."}
            
            # Analiz
            genai.configure(api_key=api_anahtari)
            model = genai.GenerativeModel('gemini-2.5-pro')
            
            analiz_prompt = f"""
            Aşağıdaki web sitesi içeriğini analiz et ve Türkçe olarak özetle:
            
            BAŞLIK: {baslik}
            URL: {url}
            İÇERİK (İlk 4000 karakter): {icerik[:4000]}
            
            Lütfen şu bilgileri ver:
            1. Web sitesinin ana konusu
            2. Önemli bilgiler (3-5 madde)
            3. İçerik türü (Blog, haber, eğitim vb.)
            """
            
            analiz_response = model.generate_content(analiz_prompt)
            
            self.mevcut_url = url
            self.mevcut_icerik = icerik
            
            return {
                "baslik": baslik,
                "icerik": icerik[:2000], # Önizleme için kısa içerik
                "url": url,
                "analiz": analiz_response.text,
                "alınma_tarihi": datetime.datetime.now(),
                "tam_icerik": icerik  # Tam içerik sonraki sorular için
            }
            
        except Exception as e:
            return {"hata": str(e)}

    def web_icerik_analiz(self, web_verisi: Dict[str, Any], soru: str, api_anahtari: str = "") -> str:
        """Web içeriği hakkında soru sor"""
        if not api_anahtari:
            return "Lütfen önce API anahtarınızı giriniz."
        
        try:
            genai.configure(api_key=api_anahtari)
            model = genai.GenerativeModel('gemini-2.5-pro')
            
            
            tam_icerik = web_verisi.get('tam_icerik', web_verisi.get('icerik', ''))
            
            prompt = f"""
            Aşağıdaki web sitesi içeriğini kullanarak soruyu Türkçe olarak yanıtla:
            
            WEB SİTESİ: {web_verisi['baslik']}
            URL: {web_verisi['url']}
            İÇERİK: {tam_icerik}
            
            SORU: {soru}
            
            Lütfen detaylı ve açıklayıcı bir cevap ver. Web sitesindeki bilgilere dayanarak yanıt oluştur.
            """
            
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Analiz hatası: {str(e)}"

# ==================== RAG CHATBOT MODÜLÜ ====================
class RAGIsleyici:
    def __init__(self):
        self.api_anahtari = None
        self.embeddings = None  

    
    def _load_embeddings(self):
        """Embedding modelini sadece gerektiğinde (butona basılınca) yükler."""
        if self.embeddings is None:
            try:
                # --- DEĞİŞTİ ---
                # Kullanıcıya indirme başladığını haber ver
                st.info("Embedding modeli Hugging Face'den indiriliyor... (İlk çalıştırmada uzun sürebilir)")
                
                
                self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
                
                st.success("Embedding modeli başarıyla yüklendi.")
            except Exception as e:
                
                st.error(f"Embedding modeli yüklenemedi: {e}. Lütfen internet bağlantınızı kontrol edin veya model adını (KULLANICI_ADINIZ/MODEL_ADINIZ) doğru yazdığınızdan emin olun.")
                return False
        return True

    def ayarla(self, api_anahtari: str):
        """API anahtarını ayarlar."""
        self.api_anahtari = api_anahtari

    def _get_pdf_text(self, pdf_docs):
        """Yüklenen PDF dosyalarından metinleri okur."""
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PyPDF2.PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def _get_text_chunks(self, text):
        """Metni işlenebilir küçük parçalara (chunk) ayırır."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        return chunks

    # YENİ METOD (Klasörden PDF'leri otomatik okuyan)
    def initialize_vector_store(self, pdf_folder_path="rag_pdfs"):
        """
        Belirtilen klasördeki TÜM PDF'leri okur, chunk'lar oluşturur,
        embedding'lerini alır ve FAISS vektör veritabanını oluşturur.
        Sonucu session_state'de saklar.
        """
        # Eğer vektör deposu zaten oluşturulmuşsa, tekrar yapma
        if "rag_vector_store" in st.session_state:
            return True

        if not self.api_anahtari:
            st.error("API anahtarı ayarlanmamış. RAG başlatılamıyor.")
            return False
        
        # Embedding modelini yükle (diskten)
        if not self._load_embeddings():
            return False # Yükleme başarısız olursa dur

        pdf_files = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
        
        if not pdf_files:
            st.error(f"'{pdf_folder_path}' klasöründe hiç PDF dosyası bulunamadı.")
            return False

        with st.spinner(f"⏳ {len(pdf_files)} PDF dosyası işleniyor ve RAG veritabanı oluşturuluyor..."):
            all_text = ""
            for pdf_path in pdf_files:
                try:
                    # PDF dosyalarını 'rb' (read binary) modunda aç
                    with open(pdf_path, "rb") as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        for page in pdf_reader.pages:
                            all_text += page.extract_text() or "" # Eğer sayfa boşsa hata vermesin
                except Exception as e:
                    st.warning(f"'{os.path.basename(pdf_path)}' okunurken hata: {e}")
            
            if not all_text.strip():
                 st.error("PDF dosyalarından metin okunamadı.")
                 return False

            text_chunks = self._get_text_chunks(all_text)
            
            # Vektör veritabanını oluştur ve session'da sakla
            try:
                vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
                st.session_state.rag_vector_store = vector_store 
                st.success("✅ RAG Chatbot veritabanı başarıyla oluşturuldu!")
                return True
            except Exception as e:
                st.error(f"Vektör veritabanı oluşturulurken hata: {e}")
                return False


    def _get_conversational_chain(self):
    	"""Soru-cevap zincirini (QA Chain) oluşturur."""

    	# 1. MAP (HARİTALAMA) PROMPT'U
    	# (Her bir belge parçasına ayrı ayrı uygulanır)
    	map_prompt_template = """
    	Aşağıdaki metin parçasını analiz et ve şu soruya cevap verebilecek kısımları özetle:
    	SORU: {question}
   	 METİN: {context}

   	Sadece soruyla ilgili TÜRKÇE özeti ver:
    	"""
   	map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["context", "question"])

    	# 2. REDUCE (BİRLEŞTİRME) PROMPT'U
    	# (Tüm özetler toplanır ve bu prompt ile tek bir cevap oluşturulur)
    	combine_prompt_template = """
    	Aşağıdaki metin özetlerini kullanarak şu soruyu kapsamlı bir şekilde TÜRKÇE yanıtla:
    	SORU: {question}
    	ÖZETLER: {context}
	
    	Kapsamlı Cevap:
    	"""
    	combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["context", "question"])

    	# MODEL TANIMLAMASI
    	model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3, google_api_key=self.api_anahtari)

    	# ZİNCİRİ DOĞRU PARAMETRELERLE YÜKLE
    	chain = load_qa_chain(
        	model, 
        	chain_type="map_reduce", 
        	question_prompt=map_prompt,     # Bu, "map" adımı için kullanılır
        	combine_prompt=combine_prompt   # Bu, "reduce" adımı için kullanılır
    	)
    	return chain

    def user_input(self, user_question):
        """Kullanıcının sorusunu alır ve RAG pipeline'ını çalıştırır."""
        if "rag_vector_store" not in st.session_state:
            st.warning("Lütfen önce bir PDF dosyası yükleyip işleyin.")
            return

        vector_store = st.session_state.rag_vector_store
        
        # Kullanıcının sorusuna en benzer doküman parçalarını veritabanından bulmak
        docs = vector_store.similarity_search(user_question, k=2)
        
        chain = self._get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]

# ==================== ANA UYGULAMA SINIFI ====================
class AkilliOgrenciAsistani:
    def __init__(self):
        self.kullanici_adi = ""
        self.dersler = []
        self.haftalik_plan = {}
        self.chat_gecmisi = []
        self.api_anahtari = ""
        self.motivasyon_sistemi = MotivasyonSistemi()
        self.pdf_isleyici = PDFIsleyici()
        self.web_analiz = WebAnaliz()
        self.hedef_takip = HedefTakipSistemi()
        self.ogrenme_analitigi = OgrenmeAnalitigi()
        self.rag_isleyici = RAGIsleyici() # <--- YENİ EKLENDİ

    def gemini_ayarla(self, api_anahtari: str):
        self.api_anahtari = api_anahtari
        genai.configure(api_key=api_anahtari)
        self.rag_isleyici.ayarla(api_anahtari) # <--- YENİ EKLENDİ

    def ders_ekle(self, ders_adi: str, zorluk_seviyesi: str = "orta"):
        self.dersler.append({"adi": ders_adi, "zorluk": zorluk_seviyesi, "eklenme_tarihi": datetime.datetime.now()})

    def haftalik_plan_olustur(self) -> Dict[str, Any]:
        if not self.dersler:
            return {"hata": "Önce ders ekleyiniz."}
        plan = {}
        gunler = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
        for i, gun in enumerate(gunler):
            ders_index = i % len(self.dersler)
            plan[gun] = {"ders": self.dersler[ders_index]["adi"], "sure": "2 saat", "konu": f"{self.dersler[ders_index]['adi']} temel konular", "zorluk": self.dersler[ders_index]["zorluk"]}
        self.haftalik_plan = plan
        return plan

    def bugun_ne_calismali(self) -> str:
        if not self.haftalik_plan:
            return "Önce haftalık plan oluşturunuz."
        gunler = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
        bugun_index = datetime.datetime.now().weekday()
        bugun_gun = gunler[bugun_index]
        if bugun_gun in self.haftalik_plan:
            plan = self.haftalik_plan[bugun_gun]
            return f"🎯 **Bugün ({bugun_gun}) için önerim:**\n\n• **Ders:** {plan['ders']}\n• **Süre:** {plan['sure']}\n• **Konu:** {plan['konu']}\n• **Zorluk:** {plan['zorluk']}"
        return "Bugün için planlanmış ders bulunmuyor. Dinlenme günü! 😊"

    def chat_gecmisi_kaydet(self, kullanici: str, mesaj: str, cevap: str):
        self.chat_gecmisi.append({"kullanici": kullanici, "mesaj": mesaj, "cevap": cevap, "zaman": datetime.datetime.now()})
        if len(self.chat_gecmisi) > 100:
            self.chat_gecmisi = self.chat_gecmisi[-100:]

    def gemini_sohbet(self, mesaj: str) -> str:
        if not self.api_anahtari:
            return "Lütfen önce API anahtarınızı giriniz."
        try:
            genai.configure(api_key=self.api_anahtari)
            model = genai.GenerativeModel('gemini-2.5-pro')
            context = "\n".join([f"Kullanıcı: {chat['mesaj']}\nAsistan: {chat['cevap']}" for chat in self.chat_gecmisi[-5:]])
            prompt = f"Sen bir akıllı öğrenci asistanısın. Samimi, arkadaşça, eğlenceli, komik ve motive edici bir dil kullan.\n{context}\nKullanıcı: {mesaj}\nAsistan:"
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Sohbet hatası: {str(e)}"

# ==================== GİRİŞ EKRANI ====================
def show_login_screen():
    # --- YARDIMCI FONKSİYON ---
    def create_animated_title(line1, line2):
        html = '<div class="hero-text">'
        html += '<div class="line">'
        
        # Kelimeleri boşluğa göre ayır
        words = line1.split(' ')
        char_count = 0
        
        for word_index, word in enumerate(words):
            # Kelimenin harflerini animasyonlu ekle
            for i, char in enumerate(word):
                delay = (char_count + i) * 0.06
                html += f'<span class="letter" style="animation-delay: {delay}s;">{char}</span>'
            
            char_count += len(word)
            
            # Kelimeler arasına animasyonlu bir boşluk ekle (&nbsp; kullanarak)
            if word_index < len(words) - 1:
                delay = char_count * 0.06
                # CSS'de inline-block olduğu için çökmemesi için &nbsp; kullanıyoruz
                html += f'<span class="letter" style="animation-delay: {delay}s;">&nbsp;</span>'
                char_count += 1 # Boşluk karakterini de say
        
        html += '</div>'
        html += '<div class="line">'
        # base_delay'i toplam karakter sayısına (boşluk dahil) göre ayarla
        base_delay = char_count * 0.06
        for i, char in enumerate(line2):
            delay = base_delay + (i * 0.06)
            html += f'<span class="letter" style="animation-delay: {delay}s;">{char}</span>'
        html += '</div>'
        html += '</div>'
        return html

    # --- YÜKLEME ARAYÜZÜNÜN HTML ve CSS KODLARI ---
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Inter:wght@400&display=swap');
    
    .stApp {
        background-color: #0a0a0f;
    }
    
    /* YÜKLEME ARAYÜZÜ (OVERLAY) */
    .loading-overlay {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background-color: rgba(10, 10, 15, 0.85); backdrop-filter: blur(5px);
        z-index: 9999; display: flex; justify-content: center; align-items: center;
        flex-direction: column; opacity: 0; animation: fadeInOverlay 0.5s ease-out forwards;
    }
    @keyframes fadeInOverlay { to { opacity: 1; } }
    
    /* ---DUYGUSAL ROBOT ANİMASYONU --- */
    .loading-robot-container {
        width: 150px; height: 150px; margin-bottom: 20px;
    }

    .robot-path {
        stroke: #a8b8ff; stroke-width: 4; fill: none;
        stroke-linecap: round; stroke-linejoin: round;
        filter: drop-shadow(0 0 5px rgba(168, 184, 255, 0.9));
    }
    
    /* Ana çizim animasyonu (kafa, anten, düz ağız) */
    .robot-draw {
        stroke-dasharray: 400; stroke-dashoffset: 400;
        animation: draw 2.5s ease-out forwards;
    }
    
    .robot-feature { /* Gözler ve anten topu */
        fill: #a8b8ff; stroke: none; opacity: 0;
        animation: fadeInFeature 0.5s ease-out forwards;
        animation-delay: 2s;
    }

    /* Düz ağız: HER ZAMAN çizilir ve sonra kaybolur */
    .robot-mouth-straight {
        animation: draw 2.5s ease-out forwards, fadeOut 0.3s ease-in forwards;
        animation-delay: 0s, 2.8s; /* 2.8s sonra kaybolmaya başla */
    }

    /* Gülümseyen ve üzgün ağızlar başlangıçta görünmez */
    .robot-mouth-smile, .robot-mouth-sad {
        stroke-dasharray: 40; stroke-dashoffset: 40;
    }
    
    /* BAŞARILI DURUM: success sınıfı eklenince bu animasyon tetiklenir */
    .success .robot-mouth-smile {
        animation: drawEmotion 1s ease-out forwards;
        animation-delay: 3s; /* Düz ağız kaybolduktan sonra başla */
    }
    
    /* HATALI DURUM: error sınıfı eklenince bu animasyon tetiklenir */
    .error .robot-mouth-sad {
        stroke: #ff6b6b; /* Üzgün ağzın rengini kırmızımsı yapalım */
        filter: drop-shadow(0 0 5px #ff6b6b);
        animation: drawEmotion 1s ease-out forwards;
        animation-delay: 3s;
    }

    @keyframes draw { to { stroke-dashoffset: 0; } }
    @keyframes fadeInFeature { to { opacity: 1; } }
    @keyframes fadeOut { to { opacity: 0; stroke-dashoffset: 400; } }
    @keyframes drawEmotion { to { stroke-dashoffset: 0; } }
    /* --- GÜNCELLEME SONU --- */

    .loading-text {
        font-family: 'Inter', sans-serif; color: rgba(255, 255, 255, 0.8);
        font-size: 1rem; letter-spacing: 0.2rem; text-transform: uppercase;
        animation: pulseText 2s ease-in-out infinite;
    }
    @keyframes pulseText { 0%, 100% { opacity: 0.7; } 50% { opacity: 1; } }
    
    /* MEVCUT ANİMASYONLARINIZ DEĞİŞMEDİ */
    .hero-text { font-family: 'Orbitron', sans-serif; font-size: 4.5rem; font-weight: 700; text-align: center; letter-spacing: 0.5rem; margin: 1.5rem 0 1rem 0; line-height: 1.2; text-transform: uppercase; color: #e5e5e5; display: flex; justify-content: center; flex-direction: column; align-items: center; perspective: 500px; }
    .letter { display: inline-block; opacity: 0; transform: translateY(-40px) rotateX(90deg); text-shadow: 0 0 8px rgba(255, 255, 255, 0.5), 0 0 20px rgba(102, 126, 234, 0.7); animation: dropAndShine 1s cubic-bezier(0.68, -0.55, 0.27, 1.55) forwards; }
    @keyframes dropAndShine { from { opacity: 0; transform: translateY(-40px) rotateX(90deg) scale(1.5); } to { opacity: 1; transform: translateY(0) rotateX(0deg) scale(1.0); } }
    .hero-subtitle-container { display: flex; justify-content: center; align-items: center; margin-bottom: 2rem; opacity: 0; animation: fadeIn 1.5s ease-out forwards; animation-delay: 2.0s; }
    .hero-subtitle { font-family: 'Inter', sans-serif; font-size: 0.9rem; color: rgba(255,255,255,0.7); letter-spacing: 0.3rem; text-transform: uppercase; margin: 0 15px; white-space: nowrap; }
    .ekg-line-container { width: 120px; height: 50px; }
    .ekg-line-container.right { transform: scaleX(-1); }
    .ekg-path { stroke: #a8b8ff; stroke-width: 2.5; fill: none; stroke-dasharray: 242; stroke-dashoffset: 242; animation: drawEkg 3s ease-in-out infinite; filter: drop-shadow(0 0 4px rgba(168, 184, 255, 0.8)); }
    @keyframes drawEkg { 0% { stroke-dashoffset: 242; } 40% { stroke-dashoffset: 0; } 60% { stroke-dashoffset: 0; } 100% { stroke-dashoffset: -242; } }
    @keyframes fadeIn { to { opacity: 1; } }
    [data-testid="stForm"], [data-testid="stExpander"] { opacity: 0; animation: formFadeInUp 1s ease-out forwards; animation-delay: 3.0s; }
    @keyframes formFadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    </style>
    """, unsafe_allow_html=True)
    
    # --- SESSION STATE KONTROLÜ ---
    if st.session_state.get('loading', False):
        
        # 1. API ANAHTARINI TEST ET
        api_anahtari = st.session_state.get('login_api_anahtari', '')
        auth_status = "error"
        if api_anahtari:
            try:
                genai.configure(api_key=api_anahtari)
                next(genai.list_models())
                auth_status = "success"
            except Exception:
                auth_status = "error"
        
        # 2. TEST SONUCUNA GÖRE DOĞRU HTML'İ OLUŞTURMAK
        loading_html = f"""
        <div class="loading-overlay">
            <div class="loading-robot-container {auth_status}">
                <svg viewBox="0 0 100 100">
                    <path class="robot-path robot-draw" d="M 20 40 Q 15 40, 15 45 V 85 Q 15 90, 20 90 H 80 Q 85 90, 85 85 V 45 Q 85 40, 80 40 H 20" />
                    <path class="robot-path robot-draw" d="M 50 40 V 20" />
                    <path class="robot-path robot-mouth-straight" d="M 35 78 H 65" />
                    <path class="robot-path robot-mouth-smile" d="M 35 78 Q 50 90, 65 78" />
                    <path class="robot-path robot-mouth-sad" d="M 35 82 Q 50 70, 65 82" />
                    <circle class="robot-feature" cx="50" cy="15" r="5" />
                    <circle class="robot-feature" cx="35" cy="60" r="7" />
                    <circle class="robot-feature" cx="65" cy="60" r="7" />
                </svg>
            </div>
            <div class="loading-text">DOĞRULANIYOR...</div>
        </div>
        """
        st.markdown(loading_html, unsafe_allow_html=True)
        
        # 3. ANİMASYONUN BİTMESİNİ BEKLE
        time.sleep(4.5)

        # 4. SONUCA GÖRE YÖNLENDİRME YAP
        if auth_status == "success":
            kullanici_adi = st.session_state.get('login_kullanici_adi', '')
            st.session_state.logged_in = True; st.session_state.kullanici_adi = kullanici_adi; st.session_state.api_anahtari = api_anahtari
            if 'asistan' not in st.session_state: st.session_state.asistan = AkilliOgrenciAsistani()
            st.session_state.asistan.kullanici_adi = kullanici_adi; st.session_state.asistan.gemini_ayarla(api_anahtari)
            st.session_state.loading = False; st.rerun()
        else:
            st.session_state.loading = False
            st.session_state.login_error = "❌ Geçersiz API Anahtarı! Robot üzüldü, lütfen kontrol et."
            st.rerun()

    else:
        animated_title_html = create_animated_title("AKILLI ÖĞRENCİ", "ASİSTANI")
        st.markdown(animated_title_html, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="hero-subtitle-container">
            <div class="ekg-line-container left"><svg viewBox="0 0 120 50"><path class="ekg-path" d="M0 25 H 30 L 35 15 L 45 35 L 50 22 L 55 28 L 60 25 H 120" /></svg></div>
            <div class="hero-subtitle">GELECEĞE HAZIRLANMAK İÇİN TASARLANDI</div>
            <div class="ekg-line-container right"><svg viewBox="0 0 120 50"><path class="ekg-path" d="M0 25 H 30 L 35 15 L 45 35 L 50 22 L 55 28 L 60 25 H 120" /></svg></div>
        </div>
        """, unsafe_allow_html=True)
        
        if 'login_error' in st.session_state and st.session_state.login_error:
            st.error(st.session_state.login_error); st.session_state.login_error = None

        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            with st.form("login_form"):
                kullanici_adi = st.text_input("ADI", placeholder="Adınızı giriniz")
                api_anahtari = st.text_input("API ANAH. TARI", type="password", placeholder="API keyi giriniz")
                submitted = st.form_submit_button("GİRİŞ YAP")
                if submitted:
                    st.session_state.login_kullanici_adi = kullanici_adi; st.session_state.login_api_anahtari = api_anahtari
                    st.session_state.loading = True; st.rerun()
                
            with st.expander("🔑 API Anahtarı Nasıl Alınır?"):
                st.write("1. Google AI Studio'ya gidin"); st.write("2. Create API key seçeneği ile API Key oluşturun"); st.write("3. Oluşturduğunuz API Keyi gerekli alana girin")
# ==================== ÇIKIŞ ANİMASYONU EKRANI ====================
def show_logout_animation():
    """Çıkış yaparken SİDEBAR DAHİL tüm ekranı kaplayan, ortalı, hızlı robot animasyonu."""
    
    kullanici_adi = st.session_state.get('kullanici_adi', 'Kullanıcı')
    
    logout_html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400&family=Orbitron:wght@700&display=swap');
    
    /* --- DÜZELTME 1: SİDEBAR'I ARKADA BIRAKMA --- */
    /* Çıkış animasyonu aktifken, sidebar'ı overlay'in arkasına it */
    [data-testid="stSidebar"] {{
        z-index: 0 !important;
    }}
    /* --- DÜZELTME SONU --- */

    .logout-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        
        /* İstediğin yarı-şeffaf koyu renk (Tam siyah değil, arka plan okunmuyor) */
        background-color: rgba(10, 10, 15, 0.95); 
        backdrop-filter: blur(5px);

        /* --- DÜZELTME 2: Z-INDEX GARANTİSİ --- */
        /* Sidebar'dan (ve diğer her şeyden) üstte olmasını garanti et */
        z-index: 99999;
        /* --- DÜZELTME SONU --- */

        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column; 
        opacity: 0; 
        animation: fadeInOverlay 0.3s ease-out forwards;
    }}
    @keyframes fadeInOverlay {{ to {{ opacity: 1; }} }}

    .robot-container {{
        width: 150px;
        height: 150px;
        margin-bottom: 20px;
        filter: drop-shadow(0 0 10px rgba(168, 184, 255, 0.7));
    }}

    /* Hızlandırılmış animasyonlar (Aynı) */
    .robot-path {{
        stroke: #a8b8ff; stroke-width: 4; fill: none;
        stroke-linecap: round; stroke-linejoin: round;
        stroke-dasharray: 400;
        stroke-dashoffset: 400;
        animation: draw 1s ease-out 0.2s forwards;
    }}
    .robot-feature {{ 
        fill: #a8b8ff; stroke: none; opacity: 0;
        animation: fadeInFeature 0.5s ease-out 1.3s forwards;
    }}
    .robot-mouth-sad {{
        stroke-dasharray: 40; stroke-dashoffset: 40;
        animation: draw 0.5s ease-out 1.5s forwards;
    }}
    #waving-antenna {{
        transform-origin: 50px 40px; 
        animation: waveAntenna 1.5s ease-in-out 1.8s infinite;
    }}
    
    @keyframes draw {{ to {{ stroke-dashoffset: 0; }} }}
    @keyframes fadeInFeature {{ to {{ opacity: 1; }} }}
    @keyframes waveAntenna {{
        0% {{ transform: rotate(0deg); }}
        25% {{ transform: rotate(20deg); }}
        75% {{ transform: rotate(-20deg); }}
        100% {{ transform: rotate(0deg); }}
    }}

    .logout-text {{
        font-family: 'Orbitron', sans-serif; 
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.5rem; 
        letter-spacing: 0.1rem;
        text-transform: uppercase;
        margin-top: 25px;
        opacity: 0;
        animation: fadeInFeature 1s ease-out 0.5s forwards;
    }}
    </style>
    
    <div class="logout-overlay">
        <div class="robot-container">
            <svg viewBox="0 0 100 100">
                <path class="robot-path" d="M 20 40 Q 15 40, 15 45 V 85 Q 15 90, 20 90 H 80 Q 85 90, 85 85 V 45 Q 85 40, 80 40 H 20" />
                <circle class="robot-feature" cx="35" cy="60" r="7" />
                <circle class="robot-feature" cx="65" cy="60" r="7" />
                <path class="robot-path robot-mouth-sad" d="M 35 82 Q 50 70, 65 82" />
                <g id="waving-antenna">
                    <path class="robot-path" d="M 50 40 V 20" />
                    <circle class="robot-feature" cx="50" cy="15" r="5" />
                </g>
            </svg>
        </div>
        <div class="logout-text">GÜLE GÜLE, {kullanici_adi}!</div>
    </div>
    """
    
    st.markdown(logout_html, unsafe_allow_html=True)
    
    time.sleep(3.0) 
    
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    st.rerun()

# ==================== ANA UYGULAMA ====================
def show_main_app():
    
    # --- ÇIKIŞ ANİMASYONU KONTROLÜ ---
    if st.session_state.get('logging_out', False):
        show_logout_animation()
        return # Animasyon fonksiyonu çalışırken alttaki kodun çalışmasını engeller
    

    # --- Animasyon Gösterme Mantığı ---
    if 'show_success_animation' in st.session_state and st.session_state.show_success_animation:
        icon = "✓"
        if st.session_state.show_success_animation == 'hedef_tamamla':
            icon = "🏆"
        
        st.markdown(f"""
            <div class="success-animation-overlay">
                <div class="icon-container">
                    <span class="icon">{icon}</span>
                </div>
            </div>
            <style>
                .success-animation-overlay {{
                    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
                    z-index: 99999; display: flex; justify-content: center; align-items: center;
                    pointer-events: none; animation: fade-out-overlay 2s ease-out forwards;
                }}
                .icon-container {{
                    background-color: rgba(30, 30, 45, 0.85); backdrop-filter: blur(5px);
                    border-radius: 50%; width: 120px; height: 120px; display: flex;
                    justify-content: center; align-items: center;
                    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
                    animation: pop-in 0.5s cubic-bezier(0.68, -0.55, 0.27, 1.55) forwards;
                }}
                .icon {{
                    font-size: 60px; color: #a8b8ff; text-shadow: 0 0 15px rgba(168, 184, 255, 0.8);
                }}
                @keyframes pop-in {{ 0% {{ transform: scale(0.5); opacity: 0; }} 100% {{ transform: scale(1); opacity: 1; }} }}
                @keyframes fade-out-overlay {{ 0% {{ opacity: 1; }} 80% {{ opacity: 1; }} 100% {{ opacity: 0; }} }}
            </style>
        """, unsafe_allow_html=True)
        
        time.sleep(2)
        st.session_state.show_success_animation = None
        st.rerun()

    # --- Fonksiyonun geri kalanı ---
    if 'asistan' not in st.session_state:
        st.session_state.asistan = AkilliOgrenciAsistani()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    asistan = st.session_state.asistan

    # --- KART OLUŞTURUCU YARDIMCI FONKSİYON ---
    def create_metric_card(icon: str, title: str, value: Any, color: str):
        html_content = f"""
        <div class="metric-card">
            <div class="metric-card-icon" style="background: linear-gradient(135deg, {color} 0%, {color}99 100%); color: white; text-shadow: 0 0 10px #FFFFFF99;">
                <i class="bi {icon}"></i>
            </div>
            <div class="metric-card-content">
                <h3>{value}</h3>
                <p>{title}</p>
            </div>
        </div>
        """
        return html_content
    
    # --- BOOTSTRAP İKON CDN LİNKİ VE ORİJİNAL STİLLER ---
    st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%); }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #0f0f0f 100%); border-right: 1px solid rgba(255,255,255,0.1); }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: white !important; font-family: 'Inter', sans-serif; }
        h1 { color: white !important; font-family: 'Inter', sans-serif !important; font-weight: 700 !important; letter-spacing: 0.05rem !important; background: linear-gradient(135deg, #ffffff 0%, #a8b8ff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        h2, h3 { color: rgba(255,255,255,0.9) !important; font-family: 'Inter', sans-serif !important; font-weight: 600 !important; }
        
        /* Sekmeleri DAHA FAZLA daraltıyoruz */
        .stTabs [data-baseweb="tab"] { 
            background: transparent; 
            border: 1px solid rgba(255,255,255,0.1); 
            color: rgba(255,255,255,0.7); 
            border-radius: 8px; /* Kenar yuvarlaklığını biraz azalttık */
            
            /* --- DAHA FAZLA KÜÇÜLTME --- */
            padding: 8px 12px !important; /* İç boşluğu İYİCE azalttık (10px 16px idi) */
            font-size: 0.85rem !important; /* Yazı boyutunu BİRAZ DAHA küçülttük (0.9rem idi) */
            /* --- KÜÇÜLTME SONU --- */
            
            font-weight: 500; 
            transition: all 0.3s; 
            white-space: nowrap; /* Yazıların alta kaymasını engelle */
        }

        .stTabs [data-baseweb="tab"] { background: transparent; border: 1px solid rgba(255,255,255,0.1); color: rgba(255,255,255,0.7); border-radius: 10px; padding: 12px 24px; font-weight: 500; transition: all 0.3s; }
        .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white !important; border-color: transparent; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); }
        .stTextInput input, .stTextArea textarea, .stSelectbox select, .stNumberInput input { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.15) !important; color: white !important; border-radius: 10px !important; font-family: 'Inter', sans-serif !important; }
        .stTextInput input:focus, .stTextArea textarea:focus { border-color: #667eea !important; box-shadow: 0 0 0 1px #667eea !important; }
        .stTextInput label, .stTextArea label, .stSelectbox label, .stNumberInput label { color: rgba(255,255,255,0.8) !important; font-weight: 500 !important; font-size: 0.9rem !important; }
        .stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 0.6rem 1.5rem !important; font-weight: 600 !important; font-family: 'Inter', sans-serif !important; transition: all 0.3s !important; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important; }
        .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important; }
        [data-testid="stSidebar"] .stButton > button { width: 100%; background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.15) !important; box-shadow: none !important; }
        [data-testid="stSidebar"] .stButton > button:hover { background: rgba(255,255,255,0.1) !important; border-color: rgba(255,255,255,0.3) !important; }
        [data-testid="stMetricValue"] { color: white !important; font-size: 2rem !important; font-weight: 700 !important; }
        [data-testid="stMetricLabel"] { color: rgba(255,255,255,0.7) !important; }
        .metric-card {
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            display: flex;
            align-items: center;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            height: 100%; 
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
            border-color: rgba(255, 255, 255, 0.2);
        }
        .metric-card-icon {
            font-size: 2.2rem; 
            margin-right: 15px;
            padding: 15px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 60px; 
            height: 60px; 
        }
        .metric-card-content {
            flex-grow: 1;
        }
        .metric-card-content h3 { 
            font-size: 2rem;
            font-weight: 700;
            color: #FFFFFF;
            margin: 0;
            line-height: 1;
            -webkit-text-fill-color: #FFFFFF; 
        }
        .metric-card-content p { 
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.7);
            margin: 0;
            margin-top: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.kullanici_adi}")
        st.markdown("---")
        hedef_stats = asistan.hedef_takip.hedef_istatistikleri(asistan.kullanici_adi)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Toplam Ders", len(asistan.dersler))
        with col2:
            st.metric("Aktif Hedefler", f"{hedef_stats['aktif_hedef']}")
        
        st.markdown("##### 🏆 Başarı Oranı")
        tamamlanma_orani_int = int(hedef_stats['tamamlanma_orani'])
        st.progress(tamamlanma_orani_int, text=f"%{tamamlanma_orani_int}")
        hatirlatmalar = asistan.hedef_takip.hatirlatma_kontrol(asistan.kullanici_adi)
        if hatirlatmalar:
            st.markdown("---")
            st.markdown("### 🔔 Hatırlatmalar")
            for hatirlatma in hatirlatmalar[:3]:
                st.warning(hatirlatma)
        st.markdown("---")
        st.markdown("### 🎯 Hızlı Erişim")
        
        if st.button("💪 Motivasyon Ver", key="sidebar_motivasyon"):
            st.success(asistan.motivasyon_sistemi.rastgele_motivasyon())
        
        st.markdown("---")
        
        # ---BUTON MANTIĞI ---
        if st.button("🚪 Çıkış Yap", key="sidebar_cikis"):
            # Sadece durumu ayarla ve yeniden çalıştır
            st.session_state.logging_out = True
            st.rerun()
        
        
              

    st.title(f"🎓 Hoş Geldin, {st.session_state.kullanici_adi}!")
    st.markdown("##### Bugün hangi konuda sana yardımcı olabilirim?")
    st.markdown("---")
    
    # Emojili sekme başlıkları
    tab_labels = [
        "💬 Sohbet", "📚 Ders Planlama", "🎯 Hedeflerim", "📊 Öğrenme Analitiği", 
        "📊 Görsel Pano", "📄 PDF İşleme", "🌐 Web Analiz",
        "🤖 RAG Chatbot"  # <--- YENİ SEKME EKLENDİ
    ]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tab_labels) # <--- tab8 EKLENDİ

    
    
    with tab1:
        st.markdown("### 💬 Akıllı Asistan ile Sohbet Et")
        st.markdown("*Merak ettiklerin,derslerin, hedeflerin ve çalışma alışkanlıkların hakkında sohbet ederek sana özel tavsiyeler alabilirsin.*"); st.markdown("")
        chat_container = st.container()
        with chat_container:
            for chat in st.session_state.chat_history[-10:]:
                # GÜNCELLEME: Öğrenci emojisi eklendi
                with st.chat_message("user", avatar="🧑‍🎓"): 
                    st.markdown(chat["mesaj"])
                with st.chat_message("assistant"): 
                    st.markdown(chat["cevap"])
        
        mesaj = st.chat_input("✨ Mesajını buraya yaz...", key="chat_input")
        
        if mesaj:
            # GÜNCELLEME: Öğrenci emojisi eklendi
            with st.chat_message("user", avatar="🧑‍🎓"): 
                st.markdown(mesaj)
            with st.chat_message("assistant"):
                with st.spinner("Düşünüyorum..."):
                    cevap = asistan.gemini_sohbet(mesaj)
                    st.markdown(cevap)
            asistan.chat_gecmisi_kaydet(asistan.kullanici_adi, mesaj, cevap)
            st.session_state.chat_history.append({"mesaj": mesaj, "cevap": cevap})

    with tab2:
        st.markdown("### 📚 Ders Planlama Modülü") 
        st.markdown("*Derslerini organize et ve haftalık çalışma planı oluştur*"); st.markdown("")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### ➕ Yeni Ders Ekle")
            with st.form("ders_ekle_form", clear_on_submit=True):
                ders = st.text_input("📖 Ders Adı", placeholder="Örn: Matematik, Fizik...")
                zorluk = st.selectbox("⚡ Zorluk Seviyesi", ["Kolay", "Orta", "Zor"])
                submitted = st.form_submit_button("Ders Ekle", use_container_width=True)
                if submitted and ders:
                    asistan.ders_ekle(ders, zorluk); st.success(f"✅ {ders} başarıyla eklendi!"); st.rerun()
        with col2:
            st.markdown("#### 📋 Mevcut Derslerim")
            if asistan.dersler:
                for i, d in enumerate(asistan.dersler, 1):
                    emoji = "🟢" if d['zorluk'] == "Kolay" else "🟡" if d['zorluk'] == "Orta" else "🔴"
                    st.markdown(f"{emoji} **{d['adi']}** - _{d['zorluk']}_")
            else: st.info("Henüz ders eklenmemiş. Sol taraftan ekleyebilirsin!")
        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            if st.button("📅 Haftalık Plan Oluştur", use_container_width=True, key="plan_olustur_btn"):
                if asistan.dersler:
                    plan = asistan.haftalik_plan_olustur(); st.success("✅ Haftalık planın hazır!")
                    for gun, detay in plan.items(): st.markdown(f"**{gun}:** {detay['ders']} - {detay['sure']}")
                else: st.warning("⚠️ Önce en az bir ders eklemelisin!")
        with col4:
            if st.button("🎯 Bugün Ne Çalışmalıyım?", use_container_width=True, key="bugun_ne_calis_btn"):
                oneri = asistan.bugun_ne_calismali(); st.info(oneri)

    with tab3:
        st.markdown("### 🎯 Hedef Belirle ve Takip Et")
        st.markdown("*Hedeflerini belirle, ilerlemeni takip et ve başarıya ulaş!*"); st.markdown("")
        stats = asistan.hedef_takip.hedef_istatistikleri(asistan.kullanici_adi)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: 
            st.markdown(create_metric_card("bi-bullseye", "Aktif Hedef", stats['aktif_hedef'], "#667eea"), unsafe_allow_html=True)
        with col2: 
            st.markdown(create_metric_card("bi-check-all", "Tamamlanan", stats['tamamlanan_hedef'], "#764ba2"), unsafe_allow_html=True)
        with col3: 
            st.markdown(create_metric_card("bi-list-task", "Toplam", stats['toplam_hedef'], "#2c3e50"), unsafe_allow_html=True)
        with col4: 
            st.markdown(create_metric_card("bi-trophy-fill", "Başarı Oranı", f"{stats['tamamlanma_orani']:.0f}%", "#1d976c"), unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### ➕ Yeni Hedef Ekle")
        with st.form("hedef_ekle_form", clear_on_submit=True):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1: hedef_metni = st.text_input("🎯 Hedefin", placeholder="Örn: Bu hafta 3 ders bitireceğim")
            with col2: kategori = st.selectbox("📁 Kategori", ["Genel", "Ders", "Sınav", "Proje", "Okuma", "Diğer"])
            with col3: bitis_tarihi = st.date_input("📅 Bitiş Tarihi", min_value=datetime.datetime.now().date())
            hedef_ekle_btn = st.form_submit_button("🎯 Hedef Ekle", use_container_width=True)
            if hedef_ekle_btn and hedef_metni:
                asistan.hedef_takip.hedef_ekle(asistan.kullanici_adi, hedef_metni, bitis_tarihi, kategori)
                st.success(f"✅ Hedef eklendi: **{hedef_metni}**")
                st.session_state.show_success_animation = 'hedef_ekle'
                st.rerun()
        st.markdown("---")
        st.markdown("#### 🎯 Aktif Hedeflerim")
        aktif_hedefler = asistan.hedef_takip.aktif_hedefler(asistan.kullanici_adi)
        if aktif_hedefler:
            for hedef in aktif_hedefler:
                with st.expander(f"{'📌' if hedef['ilerleme'] < 30 else '🔥' if hedef['ilerleme'] < 70 else '⭐'} {hedef['metin']}", expanded=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Kategori:** {hedef['kategori']}")
                        kalan = (hedef['bitis_tarihi'] - datetime.datetime.now().date()).days
                        if kalan < 0: st.error(f"⏰ **Süre Doldu!** ({abs(kalan)} gün geçti)")
                        elif kalan == 0: st.warning(f"⏰ **Bugün bitiyor!**")
                        else: st.info(f"📅 **Bitiş:** {hedef['bitis_tarihi']} ({kalan} gün kaldı)")
                        st.progress(hedef['ilerleme'] / 100); st.caption(f"İlerleme: %{hedef['ilerleme']}")
                    with col2:
                        yeni_ilerleme = st.slider("İlerleme", 0, 100, hedef['ilerleme'], key=f"slider_{hedef['id']}")
                        if st.button("💾 Güncelle", key=f"guncelle_{hedef['id']}", use_container_width=True):
                            asistan.hedef_takip.hedef_guncelle(asistan.kullanici_adi, hedef['id'], yeni_ilerleme)
                            if yeni_ilerleme >= 100:
                                st.success("🎉 Tebrikler! Hedefi tamamladın!")
                                st.session_state.show_success_animation = 'hedef_tamamla'
                            st.rerun()
                        if st.button("✅ Tamamla", key=f"tamamla_{hedef['id']}", use_container_width=True):
                            asistan.hedef_takip.hedef_tamamla(asistan.kullanici_adi, hedef['id'])
                            st.success("🎉 Harika! Hedef tamamlandı!")
                            st.session_state.show_success_animation = 'hedef_tamamla'
                            st.rerun()
                        if st.button("🗑️ Sil", key=f"sil_{hedef['id']}", use_container_width=True):
                            asistan.hedef_takip.hedef_sil(asistan.kullanici_adi, hedef['id']); st.rerun()
        else: st.info("📝 Henüz aktif hedefin yok. Yukarıdan yeni hedef ekleyebilirsin!")
        st.markdown("---")
        st.markdown("#### ✅ Tamamlanan Hedefler")
        tamamlanan = asistan.hedef_takip.tamamlanan_hedefler_listesi(asistan.kullanici_adi)
        if tamamlanan:
            with st.expander(f"🏆 {len(tamamlanan)} Hedef Tamamlandı", expanded=False):
                for hedef in tamamlanan[-5:]:
                    st.success(f"✅ **{hedef['metin']}** - _{hedef['kategori']}_")
        else: st.info("Henüz tamamlanmış hedef yok. Çalışmaya devam et!")

    with tab4:
        st.markdown("### 📊 Kişisel Öğrenme Analitiği")
        st.markdown("*AI destekli çalışma analizi ve kişiselleştirilmiş öneriler*"); st.markdown("")
        st.markdown("#### 📝 Yeni Çalışma Kaydı Ekle")
        with st.form("analitik_kayit_form", clear_on_submit=True):
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1: ders_sec = st.selectbox("📚 Ders", [d["adi"] for d in asistan.dersler] if asistan.dersler else ["Genel"])
            with col2: sure_input = st.number_input("⏱️ Süre (saat)", 0.0, 12.0, 1.0, 0.5)
            with col3: konular_input = st.text_input("📖 Konular", placeholder="Virgülle ayır: Konu1, Konu2")
            kayit_btn = st.form_submit_button("💾 Kaydı Ekle", use_container_width=True)
            if kayit_btn and ders_sec and sure_input > 0:
                konu_listesi = [k.strip() for k in konular_input.split(",") if k.strip()]
                asistan.ogrenme_analitigi.calisma_kaydet(asistan.kullanici_adi, ders_sec, sure_input, konu_listesi)
                st.success(f"✅ {ders_sec} dersi için {sure_input} saatlik çalışma kaydedildi!")
                st.session_state.show_success_animation = 'kayit_ekle'
                st.rerun()
        st.markdown("---")
        st.markdown("#### ⚡ Hızlı İçgörüler")
        hizli_oneriler = asistan.ogrenme_analitigi.hizli_oneriler(asistan.kullanici_adi)
        for oneri in hizli_oneriler: st.info(oneri)
        st.markdown("---")
        st.markdown("#### 📈 Ders Bazlı Analiz")
        genel_analiz = asistan.ogrenme_analitigi.genel_analiz(asistan.kullanici_adi)
        if genel_analiz:
            for ders, veri in genel_analiz.items():
                with st.expander(f"📚 {ders}", expanded=True):
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: 
                        st.markdown(create_metric_card("bi-clock-history", "Toplam Süre", f"{veri['toplam_sure']:.1f}h", "#fd7e14"), unsafe_allow_html=True)
                    with col2: 
                        st.markdown(create_metric_card("bi-calendar-check", "Çalışma Günü", veri['calisma_gun_sayisi'], "#20c997"), unsafe_allow_html=True)
                    with col3: 
                        st.markdown(create_metric_card("bi-book", "Konu Sayısı", veri['toplam_konu'], "#0dcaf0"), unsafe_allow_html=True)
                    with col4: 
                        durum_renk = "#28a745" if veri['durum'] == "aktif" else "#ffc107"
                        durum_ikon = "bi-check-circle-fill" if veri['durum'] == "aktif" else "bi-pause-circle-fill"
                        st.markdown(create_metric_card(durum_ikon, "Durum", f"{veri['durum'].title()}", durum_renk), unsafe_allow_html=True)
                    
                    if veri['son_calisma'] == 0: st.success("🎉 Bugün çalıştın! Harika!")
                    elif veri['son_calisma'] == 1: st.info("👍 Dün çalışmıştın, bugün de devam edebilirsin!")
                    elif veri['son_calisma'] <= 3: st.warning(f"⏰ {veri['son_calisma']} gün önce çalışmıştın, tekrar başlama zamanı!")
                    else: st.error(f"⚠️ {veri['son_calisma']} gündür çalışmıyorsun! Hemen başla!")
                    if veri.get('son_7_gun_calisma', 0) >= 5: st.success(f"🔥 Son 7 günde {veri['son_7_gun_calisma']} gün çalıştın! Muhteşem tempo!")
                    elif veri.get('son_7_gun_calisma', 0) >= 3: st.info(f"💪 Son 7 günde {veri['son_7_gun_calisma']} gün çalıştın. İyi gidiyorsun!")
        else: st.info("📝 Henüz çalışma kaydı yok. Yukarıdan ekleyerek başlayabilirsin!")
        st.markdown("---")
        st.markdown("#### 🤖 Kişiselleştirilmiş Öneriler")
        st.markdown("*Asistanın çalışma verilerini analiz ederek sana özel önerilerde bulunuyor*")
        if st.button("🤖Analizi Başlat", use_container_width=True, key="ai_analiz_btn"):
            if not genel_analiz: st.warning("⚠️Analiz için önce çalışma kayıtları eklemelisin!")
            else:
                with st.spinner("🤖Asistanın verilerini analiz ediyor ve öneriler hazırlıyor..."):
                    ai_onerileri = asistan.ogrenme_analitigi.ai_onerileri_olustur(asistan.kullanici_adi, asistan.api_anahtari)
                    st.markdown("---"); st.markdown(ai_onerileri); st.markdown("---"); st.success("✨Analizi tamamlandı!")
        if genel_analiz:
            st.markdown("---"); st.markdown("#### 📊 Genel İstatistikler")
            toplam_sure = sum([v['toplam_sure'] for v in genel_analiz.values()]); toplam_ders = len(genel_analiz)
            aktif_dersler = len([v for v in genel_analiz.values() if v['durum'] == 'aktif']); toplam_konu = sum([v['toplam_konu'] for v in genel_analiz.values()])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: 
                st.markdown(create_metric_card("bi-journals", "Takip Edilen Ders", toplam_ders, "#667eea"), unsafe_allow_html=True)
            with col2: 
                st.markdown(create_metric_card("bi-fire", "Aktif Ders", aktif_dersler, "#dc3545"), unsafe_allow_html=True)
            with col3: 
                st.markdown(create_metric_card("bi-stopwatch-fill", "Toplam Çalışma", f"{toplam_sure:.1f}h", "#fd7e14"), unsafe_allow_html=True)
            with col4: 
                st.markdown(create_metric_card("bi-card-list", "Toplam Konu", toplam_konu, "#2c3e50"), unsafe_allow_html=True)
    
    with tab5:
        st.markdown("### 📊 Görsel Öğrenme Panosu")
        st.markdown("*Son bir haftalık çalışma performansını grafiklerle incele.*"); st.markdown("")
        fig_bar, fig_line, ozet = asistan.ogrenme_analitigi.gorsel_pano_olustur(asistan.kullanici_adi)
        st.info(ozet)
        if fig_bar and fig_line:
            st.markdown("---"); st.pyplot(fig_bar); st.markdown("---"); st.pyplot(fig_line); plt.close('all')

    with tab6:
        st.markdown("### 📄 PDF Analiz ve Özet")
        st.markdown("*PDF dosyalarını yükle, analiz et, özetle ve soru sor*"); st.markdown("")
        
        st.warning("🔒 **Gizlilik Notu:** Analiz için yüklediğiniz PDF dosyası, Google'ın sunucularına gönderilecektir. Lütfen çok hassas veya kişisel veriler içeren belgeleri yüklemeyin.", icon="⚠️")
        st.markdown("---")

        pdf = st.file_uploader("📎 PDF Dosyası Seç", type="pdf", key="pdf_uploader_main_tab")
        
        if pdf:
            st.success(f"✅ PDF başarıyla yüklendi: **{pdf.name}**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📝 PDF'i Akıllı Asistan ile Özetle", use_container_width=True, key="pdf_ozet_btn"):
                    with st.spinner("🤖 PDF özenle analiz ediliyor..."):
                        ozet = asistan.pdf_isleyici.pdf_ozetle(pdf, api_anahtari=asistan.api_anahtari)
                        st.markdown("#### 📋 Özet"); st.markdown(ozet)
            with col2:
                st.markdown("#### ❓ PDF Hakkında Soru Sor")
                with st.form("pdf_soru_form"):
                    soru = st.text_input("Sorunuzu yazın", placeholder="Bu PDF'te hangi konular işleniyor?")
                    soruldu = st.form_submit_button("🤖 Asistana Sor", use_container_width=True)
                    if soruldu and soru:
                        with st.spinner("🤖 Cevap hazırlanıyor..."):
                            cevap = asistan.pdf_isleyici.pdf_soru_cevapla(pdf, soru, asistan.api_anahtari)
                            st.markdown("#### 💡 Cevap"); st.markdown(cevap)
        else:
            st.info("👆 Analiz etmek için bir PDF dosyası yükleyin")

    with tab7:
        st.markdown("### 🌐 Web Sitesi Analizi")
        st.markdown("*Herhangi bir web sitesini analiz et ve sorular sor*"); st.markdown("")
        
        st.warning("🔒 **Gizlilik Notu:** Analiz için girilen web sitesinin içeriği, Google'ın sunucularına gönderilecektir. Lütfen gizli veya erişimi kısıtlı sitelerin linklerini girmeyin.", icon="⚠️")
        st.markdown("---")
        
        url = st.text_input("🔗 Web Sitesi URL'si", placeholder="https://example.com", key="web_url_input")

        if st.button("🔍 Web Sitesini Analiz Et", use_container_width=True, key="web_analiz_btn") and url:
            with st.spinner("🤖 Web sitesi analiz ediliyor.."):
                veri = asistan.web_analiz.web_sitesi_oku(url, api_anahtari=asistan.api_anahtari)
                if "hata" not in veri:
                    st.session_state.web_data = veri; st.success(f"✅ **{veri['baslik']}**"); st.markdown("#### 🤖 Asistanın Görüşü"); st.markdown(veri['analiz'])
                    with st.expander("📄 İçerik Önizleme"): st.text(veri['icerik'][:500] + "...")
                else: st.error(f"❌ Hata: {veri['hata']}")
                
        if 'web_data' in st.session_state:
            st.markdown("---"); st.markdown("#### ❓ Web Sitesi Hakkında Soru Sor")
            with st.form("web_soru_form"):
                web_soru = st.text_input("Sorunuzu yazın", placeholder="Bu web sitesinde hangi bilgiler var?")
                web_soruldu = st.form_submit_button("🤖 Asistana Sor", use_container_width=True)
                if web_soruldu and web_soru:
                    with st.spinner("🤖 Asistanın cevabı dikkatlice hazırlıyor..."):
                        cevap = asistan.web_analiz.web_icerik_analiz(st.session_state.web_data, web_soru, asistan.api_anahtari)
                        st.markdown("#### 💡 Cevap"); st.markdown(cevap)

   # --- RAG SEKME ---
    with tab8:
        st.markdown("### 🤖 RAG Temelli PDF Chatbot")
        st.info("Bu chatbot, uygulama ile birlikte gelen PDF'lerdeki bilgilere dayanarak sorularınızı yanıtlar.", icon="📚")
        st.markdown("""
        **Nasıl Çalışır?**
        Uygulama, `rag_pdfs` klasöründeki PDF dosyalarını otomatik olarak işleyerek bir bilgi veritabanı oluşturmuştur. Sorularınız bu veritabanı kullanılarak cevaplanır.
        """)
        st.markdown("---")

        # RAG veritabanının hazır olup olmadığını KONTROL ET veya İLK KEZ OLUŞTUR
        # Bu fonksiyon, veritabanı zaten varsa True döner, yoksa oluşturmayı dener.
        rag_ready = asistan.rag_isleyici.initialize_vector_store() 

        if rag_ready:
            st.markdown("#### 💬 PDF İçeriği Hakkında Soru Sor")
            user_question = st.text_input("Sorunuzu buraya yazın:", key="rag_question_input", placeholder="Örn: Zaman yönetimi için hangi teknikler var?")
            if user_question:
                with st.spinner("Cevap aranıyor..."):
                    response = asistan.rag_isleyici.user_input(user_question)
                    st.write("### Asistanın Cevabı:")
                    st.markdown(response)
        else:
            # initialize_vector_store içinde zaten hata mesajı gösterildi.
            st.error("RAG Chatbot başlatılamadı. Lütfen yönetici ile iletişime geçin veya PDF dosyalarını kontrol edin.")

   

def main():
    st.set_page_config(page_title="Akıllı Öğrenci Asistanı", page_icon="🎓", layout="wide")
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if st.session_state.logged_in:
        show_main_app()
    else:
        show_login_screen()

if __name__ == "__main__":
    main()