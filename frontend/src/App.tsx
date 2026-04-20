import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { BatchPage } from '@/pages/BatchPage';
import { SinglePage } from '@/pages/SinglePage';

export default function App() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b">
        <div className="container flex h-14 items-center justify-between">
          <h1 className="text-lg font-semibold">Vehicle Detection</h1>
          <span className="text-xs text-muted-foreground">YOLOv8 + SAM</span>
        </div>
      </header>
      <main className="container py-6">
        <Tabs defaultValue="single">
          <TabsList>
            <TabsTrigger value="single">단일 이미지</TabsTrigger>
            <TabsTrigger value="batch">배치</TabsTrigger>
          </TabsList>
          <TabsContent value="single">
            <SinglePage />
          </TabsContent>
          <TabsContent value="batch">
            <BatchPage />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
