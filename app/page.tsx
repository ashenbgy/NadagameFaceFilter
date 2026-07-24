import FaceFilter from '@/components/FaceFilter';

export const metadata = {
  title: 'AR Face Mask Filter',
  description: 'Real-time Augmented Reality Face Mask Filter built with Next.js and MediaPipe.',
};

export default function Home() {
  return (
    <main>
      <header>
        <h1>YAKA AR</h1>
        <p className="subtitle">
          Experience a beautiful, audio-reactive digital mask that tracks your facial expressions in real time.
        </p>
      </header>

      <FaceFilter />

      <footer style={{ marginTop: '2rem', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.9rem', animation: 'fadeUp 1s cubic-bezier(0.16, 1, 0.3, 1) forwards' }}>
        <p>developed by ashenbgy</p>
      </footer>
    </main>
  );
}
