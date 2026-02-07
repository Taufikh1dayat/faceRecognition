import { useEffect, useMemo, useState } from "react";

const profile = {
  name: "Nama Anda",
  role: "Frontend Developer",
  location: "Jakarta, Indonesia",
  intro:
    "Saya membangun pengalaman digital yang rapi, cepat, dan berfokus pada pengguna. Saya suka mengubah ide menjadi produk yang siap dipakai.",
  email: "emailanda@domain.com",
  resumeUrl: "https://drive.google.com/",
  github: "https://github.com/username",
  linkedin: "https://linkedin.com/in/username",
};

const highlights = [
  {
    title: "Produk siap produksi",
    description: "Fokus pada UI yang konsisten, aksesibel, dan mudah diskalakan.",
  },
  {
    title: "Kolaborasi cepat",
    description: "Terbiasa kerja dengan desainer & backend untuk hasil terbaik.",
  },
  {
    title: "Teknologi modern",
    description: "React, Vite, Tailwind, Figma, serta integrasi API.",
  },
];

const skills = [
  "React",
  "TypeScript",
  "Vite",
  "Node.js",
  "UI/UX",
  "Figma",
  "REST API",
  "Git & GitHub",
];

const services = [
  {
    title: "Website Portfolio",
    description: "Membangun portofolio yang fokus pada personal branding dan konversi.",
  },
  {
    title: "Landing Page",
    description: "Mendesain halaman promosi untuk produk baru dengan performa tinggi.",
  },
  {
    title: "Dashboard",
    description: "Menyusun dashboard data yang informatif dan mudah dipahami.",
  },
];

const formatDate = (value) =>
  new Date(value).toLocaleDateString("id-ID", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });

export default function App() {
  const [repos, setRepos] = useState([]);
  const [status, setStatus] = useState("idle");

  const username = useMemo(() => profile.github.split("/").pop(), []);

  useEffect(() => {
    if (!username || username === "username") {
      return;
    }

    const controller = new AbortController();

    const loadRepos = async () => {
      setStatus("loading");
      try {
        const response = await fetch(
          `https://api.github.com/users/${username}/repos?per_page=6&sort=updated`,
          { signal: controller.signal }
        );
        if (!response.ok) {
          throw new Error("Gagal memuat repositori");
        }
        const data = await response.json();
        const filtered = data.filter((repo) => !repo.fork);
        setRepos(filtered.slice(0, 6));
        setStatus("success");
      } catch (error) {
        if (error.name !== "AbortError") {
          setStatus("error");
        }
      }
    };

    loadRepos();

    return () => controller.abort();
  }, [username]);

  return (
    <div className="page">
      <header className="hero">
        <nav className="nav">
          <div className="logo">{profile.name}</div>
          <div className="nav-links">
            <a href="#about">Tentang</a>
            <a href="#projects">Projects</a>
            <a href="#services">Layanan</a>
            <a href="#contact">Kontak</a>
          </div>
          <a className="button ghost" href={profile.resumeUrl}>
            Download CV
          </a>
        </nav>

        <div className="hero-content">
          <div>
            <p className="eyebrow">Available for freelance</p>
            <h1>
              Halo, saya <span>{profile.name}</span>. <br />
              {profile.role} yang siap membantu produk Anda tumbuh.
            </h1>
            <p className="subtitle">{profile.intro}</p>
            <div className="hero-actions">
              <a className="button" href={profile.github}>
                Lihat GitHub
              </a>
              <a className="button secondary" href="#projects">
                Lihat Project
              </a>
            </div>
          </div>
          <div className="hero-card">
            <div className="hero-card__top">
              <span className="badge">{profile.location}</span>
              <span className="badge ghost">Open to work</span>
            </div>
            <div className="hero-card__body">
              <h3>Ringkasan Cepat</h3>
              <ul>
                <li>3+ tahun membangun antarmuka modern</li>
                <li>Berfokus pada UI yang bersih & performa</li>
                <li>Siap kolaborasi remote maupun onsite</li>
              </ul>
              <div className="hero-card__footer">
                <a href={profile.linkedin}>LinkedIn</a>
                <a href={profile.github}>GitHub</a>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main>
        <section id="about" className="section">
          <div className="section-heading">
            <h2>Tentang Saya</h2>
            <p>
              Menciptakan pengalaman digital yang indah, fungsional, dan dapat
              diukur.
            </p>
          </div>
          <div className="grid three">
            {highlights.map((item) => (
              <article key={item.title} className="card">
                <h3>{item.title}</h3>
                <p>{item.description}</p>
              </article>
            ))}
          </div>
        </section>

        <section id="projects" className="section alt">
          <div className="section-heading">
            <h2>Project Terbaru</h2>
            <p>
              Project diambil langsung dari GitHub Anda dan selalu ter-update
              otomatis.
            </p>
          </div>
          <div className="project-grid">
            {status === "idle" && (
              <div className="card full">
                <h3>Hubungkan GitHub Anda</h3>
                <p>
                  Ganti link GitHub di file <strong>src/App.jsx</strong> agar
                  daftar project tampil otomatis.
                </p>
              </div>
            )}
            {status === "loading" && (
              <div className="card full">Memuat project terbaru...</div>
            )}
            {status === "error" && (
              <div className="card full">
                Maaf, project belum dapat dimuat. Coba cek koneksi atau nama
                pengguna GitHub Anda.
              </div>
            )}
            {status === "success" &&
              repos.map((repo) => (
                <article key={repo.id} className="card project">
                  <div>
                    <h3>{repo.name}</h3>
                    <p>{repo.description || "Tidak ada deskripsi."}</p>
                  </div>
                  <div className="project-meta">
                    <span>{repo.language || "Multi"}</span>
                    <span>Updated {formatDate(repo.updated_at)}</span>
                  </div>
                  <a className="link" href={repo.html_url}>
                    Lihat repo →
                  </a>
                </article>
              ))}
          </div>
        </section>

        <section id="services" className="section">
          <div className="section-heading">
            <h2>Layanan</h2>
            <p>Siap membantu bisnis Anda tampil lebih profesional.</p>
          </div>
          <div className="grid three">
            {services.map((service) => (
              <article key={service.title} className="card">
                <h3>{service.title}</h3>
                <p>{service.description}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="section alt">
          <div className="section-heading">
            <h2>Skills</h2>
            <p>Tools utama yang saya gunakan untuk membangun produk digital.</p>
          </div>
          <div className="skills">
            {skills.map((skill) => (
              <span key={skill}>{skill}</span>
            ))}
          </div>
        </section>

        <section id="contact" className="section">
          <div className="contact">
            <div>
              <h2>Siap Kolaborasi?</h2>
              <p>
                Kirim pesan dan ceritakan kebutuhan project Anda. Saya akan
                merespon secepatnya.
              </p>
            </div>
            <div className="contact-card">
              <p>Kontak langsung</p>
              <a className="button" href={`mailto:${profile.email}`}>
                {profile.email}
              </a>
            </div>
          </div>
        </section>
      </main>

      <footer className="footer">
        <p>© 2024 {profile.name}. Dibuat dengan React.</p>
        <div>
          <a href={profile.github}>GitHub</a>
          <a href={profile.linkedin}>LinkedIn</a>
        </div>
      </footer>
    </div>
  );
}
