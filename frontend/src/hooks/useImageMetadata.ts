import { useState, useEffect } from 'react';
import exifr from 'exifr';

export function useImageMetadata(imageSrc: string | null) {
  const [metadata, setMetadata] = useState<any>(null);
  const [address, setAddress] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    async function extract() {
      if (!imageSrc) return;
      setLoading(true);
      setAddress(null);

      try {
        const data = await exifr.parse(imageSrc, {
          tiff: true, xmp: true, gps: true, translateKeys: true,
        });
        setMetadata(data);

        // Reverse Geocoding se esistono le coordinate
        if (data?.latitude && data?.longitude) {
          const res = await fetch(
            `https://nominatim.openstreetmap.org/reverse?format=json&lat=${data.latitude}&lon=${data.longitude}&zoom=10&addressdetails=1`,
            { headers: { 'Accept-Language': 'it' } }
          );
          const geoData = await res.json();
          
          // Componiamo una stringa leggibile (Città, Paese)
          const city = geoData.address.city || geoData.address.town || geoData.address.village;
          const country = geoData.address.country;
          setAddress(city ? `${city}, ${country}` : country);
        }
      } catch (e) {
        console.warn("Errore:", e);
        setMetadata(null);
      } finally {
        setLoading(false);
      }
    }
    extract();
  }, [imageSrc]);

  return { metadata, address, loading };
}