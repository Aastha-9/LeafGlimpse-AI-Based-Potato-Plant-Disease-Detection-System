import React from 'react';

const Hero = ({ language }) => {
  const content = {
    en: {
      title: "Smart Crop Disease Detection",
      subtitle: "Upload a clear image of your plant's leaf, and our advanced AI will instantly identify diseases and provide expert remedies."
    },
    hi: {
      title: "स्मार्ट फसल रोग पहचान",
      subtitle: "अपने पौधे की पत्ती की स्पष्ट छवि अपलोड करें, और हमारा उन्नत एआई तुरंत बीमारियों की पहचान करेगा।"
    },
    mr: {
      title: "स्मार्ट पीक रोग ओळख",
      subtitle: "तुमच्या वनस्पतीच्या पानाचा स्पष्ट फोटो अपलोड करा, आमचे प्रगत AI त्वरित रोगाची ओळख करेल."
    }
  };

  const t = content[language] || content['en'];

  return (
    <section className="hero">
      <h1>{t.title}</h1>
      <p>{t.subtitle}</p>
    </section>
  );
};

export default Hero;
