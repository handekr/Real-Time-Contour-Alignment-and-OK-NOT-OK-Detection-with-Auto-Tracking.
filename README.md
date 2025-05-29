# Real-Time-Contour-Alignment-and-OK-NOT-OK-Detection-with-Auto-Tracking.

🇹🇷 Açıklama (Türkçe):
Bu sistem, Daheng endüstriyel kamerasından alınan canlı görüntüdeki nesne konturlarını analiz eder ve JSON dosyasından alınan referans kontur ile karşılaştırarak hizalama (alignment) kontrolü yapar. Eğer canlı kontur referans kontur içinde yer alıyorsa veya belirli bir mesafe toleransı içindeyse “OK”, aksi halde “NOT OK” sonucu verir.
Sistem aynı zamanda hizalama başarılı olduğunda nesnenin hareketini otomatik olarak takip eder ve merkez kaymalarına göre referans konturu dinamik olarak taşır. Görsel geri bildirimle iç/dış alanlar farklı renklerle gösterilir.

🇬🇧 Description (English):
This system captures real-time video from a Daheng industrial camera, detects object contours, and compares them to a reference shape loaded from a JSON file. It performs alignment validation by checking whether the live contour is inside or close to the reference contour. If aligned, it displays an “OK” result; otherwise, it shows “NOT OK.”
Once alignment is confirmed, the system automatically tracks the object's movement by adjusting the reference contour based on the shifting center point. Visual feedback highlights matching and mismatched areas with different overlay colors.

