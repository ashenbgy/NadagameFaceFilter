import FilterStudio from '@/components/FilterStudio';

export const metadata = {
  title: 'YAKA AR · Live Face Filters',
  description: 'Real-time face masks and AI ghost effects built with MediaPipe.',
};

export default function Home() {
  return (
    <main>
      <header>
        <h1>YAKA AR</h1>
        <p className="subtitle">
          Choose your reality. Wear the original audio-reactive mask or disappear live with AI Ghost Mode.
        </p>
      </header>

      <FilterStudio />

      <footer style={{ marginTop: '2rem', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.9rem', animation: 'fadeUp 1s cubic-bezier(0.16, 1, 0.3, 1) forwards' }}>
        <p>developed by ashenbgy</p>
      </footer>
    </main>
  );
}
