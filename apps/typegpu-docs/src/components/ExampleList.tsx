import { examplesByCategory } from '../utils/examples/exampleContent.ts';
import { exampleCategories } from '../utils/examples/types.ts';
import { ExampleLink } from './ExampleLink.tsx';

function ExampleList({ excludeTags = [] }: { excludeTags?: string[] }) {
  return (
    <>
      <nav className='box-border flex flex-col flex-1 gap-6 py-4 overflow-y-auto min-w-64'>
        {
          /* <ExampleLink key={PLAYGROUND_KEY} exampleKey={PLAYGROUND_KEY}>
          Playground
        </ExampleLink>
        <hr className="border-tameplum-100" /> */
        }
        {exampleCategories.map((category) =>
          (examplesByCategory[category.key] ?? []).map((example) => {
            if (
              example.metadata.tags?.some((tag) => excludeTags.includes(tag))
            ) {
              return null;
            }

            return (
              <ExampleLink
                key={example.key}
                exampleKey={example.key}
                isExperimental={example.metadata.tags?.includes(
                  'experimental',
                ) ?? false}
              >
                {example.metadata.title}
              </ExampleLink>
            );
          })
        )}
      </nav>
    </>
  );
}

export default ExampleList;
