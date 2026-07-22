import FaceFilter from '@/components/FaceFilter';

export const metadata = {
  title: 'AR Face Mask Filter',
  description: 'Real-time Augmented Reality Face Mask Filter built with Next.js and MediaPipe.',
};

export default function Home() {
  return (
    <main>
      <header>
        <h1>Immersive AR</h1>
        <p className="subtitle">
          Experience a beautiful, audio-reactive digital mask that tracks your facial expressions in real time.
        </p>
      </header>
      
      <FaceFilter />
      
    </main>
  );
}
