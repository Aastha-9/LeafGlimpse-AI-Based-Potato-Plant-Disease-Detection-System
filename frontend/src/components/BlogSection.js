import React, { useEffect, useState } from 'react';
import { X } from 'lucide-react';

const BlogSection = ({ language }) => {
  const [blogs, setBlogs] = useState([]);
  const [selectedBlog, setSelectedBlog] = useState(null);

  useEffect(() => {
    fetch(`/blogs?lang=${language}`)
      .then(res => res.json())
      .then(data => setBlogs(data))
      .catch(err => console.error("Failed to load blogs:", err));
  }, [language]); // Refetch when language changes

  if (blogs.length === 0) return null;

  return (
    <section className="blog-section">
      <h2 className="section-title">
        {language === "hi" ? "ब्लॉग अनुभाग" : language === "mr" ? "ब्लॉग विभाग" : "Blogs Section"}
      </h2>
      <div className="blog-grid">
        {blogs.map(blog => (
          <article key={blog.id} className="blog-card" onClick={() => setSelectedBlog(blog)} style={{cursor: 'pointer', overflow: 'hidden', display: 'flex', flexDirection: 'column'}}>
            <div style={{padding: '20px', flex: 1, display: 'flex', flexDirection: 'column'}}>
              <span className="blog-category" style={{marginBottom: '10px', display: 'inline-block'}}>{blog.category}</span>
              <h3 style={{marginBottom: '10px'}}>{blog.title}</h3>
              <p style={{marginBottom: '20px', flex: 1}}>{blog.excerpt}</p>
              <span className="read-more" style={{marginTop: 'auto', display: 'inline-block'}}>
                {language === "hi" ? "पूरा लेख पढ़ें" : language === "mr" ? "संपूर्ण लेख वाचा" : "Read Full Article"} &rarr;
              </span>
            </div>
          </article>
        ))}
      </div>

      {selectedBlog && (
        <div style={{
          position: 'fixed', top: 0, left: 0, width: '100%', height: '100%',
          background: 'rgba(0,0,0,0.5)', zIndex: 1000, display: 'flex',
          justifyContent: 'center', alignItems: 'center', padding: '20px'
        }} onClick={() => setSelectedBlog(null)}>
          <div style={{
            background: 'white', padding: '30px', borderRadius: '16px', maxWidth: '600px',
            width: '100%', position: 'relative', maxHeight: '80vh', overflowY: 'auto'
          }} onClick={e => e.stopPropagation()}>
            <button 
              onClick={() => setSelectedBlog(null)}
              style={{position: 'absolute', top: '20px', right: '20px', background: 'transparent', border: 'none', cursor: 'pointer', color: '#64748b'}}
            >
              <X size={24} />
            </button>
            <span className="blog-category" style={{marginBottom: '10px'}}>{selectedBlog.category}</span>
            <h2 style={{color: 'var(--primary-dark)', marginBottom: '20px'}}>{selectedBlog.title}</h2>
            <p style={{lineHeight: '1.8', color: 'var(--text-main)', fontSize: '1.1rem'}}>{selectedBlog.content}</p>
          </div>
        </div>
      )}
    </section>
  );
};

export default BlogSection;
