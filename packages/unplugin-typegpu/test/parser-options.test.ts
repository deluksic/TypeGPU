import { describe, expect, it } from 'vitest';
import { babelTransform, rollupTransform } from './transform.ts';

describe('[BABEL] parser options', () => {
  it('with no include option, import determines whether to run the plugin', () => {
    const codeWithImport = `\
      import tgpu from 'typegpu';
      
      const increment = tgpu['~unstable']
        .fn([])(() => {
          const x = 2+2;
        });
    `;

    expect(
      babelTransform(codeWithImport, { include: [/virtual:/] }),
    ).toMatchInlineSnapshot(`
      "import tgpu from 'typegpu';
      const increment = tgpu['~unstable'].fn([])(tgpu.__assignAst(tgpu.__removedJsImpl(), {"argNames":{"type":"identifiers","names":[]},"body":[0,[[13,"x",[1,[5,"2"],"+",[5,"2"]]]]],"externalNames":[]}, {}));"
    `);

    const codeWithoutImport = `\
      const increment = tgpu['~unstable']
        .fn([])(() => {
          const x = 2+2;
        });
    `;

    expect(
      babelTransform(codeWithoutImport, { include: [/virtual:/] }),
    ).toMatchInlineSnapshot(`
      "const increment = tgpu['~unstable'].fn([])(() => {
        const x = 2 + 2;
      });"
    `);
  });
});

describe('[ROLLUP] tgpu alias gathering', async () => {
  it('with no include option, import determines whether to run the plugin', async () => {
    const codeWithImport = `\
      import tgpu from 'typegpu';
      
      const increment = tgpu['~unstable']
        .fn([])(() => {
          const x = 2+2;
        });

      console.log(increment);
  `;

    expect(
      await rollupTransform(codeWithImport, { include: [/virtual:/] }),
    ).toMatchInlineSnapshot(`
      "import tgpu from 'typegpu';

      const increment = tgpu['~unstable']
              .fn([])(tgpu.__assignAst(tgpu.__removedJsImpl(), {"argNames":{"type":"identifiers","names":[]},"body":[0,[[13,"x",[1,[5,"2"],"+",[5,"2"]]]]],"externalNames":[]}));

            console.log(increment);
      "
    `);

    const codeWithoutImport = `\
      const increment = tgpu['~unstable']
        .fn([])(() => {
          const x = 2+2;
        });

      console.log(increment);
    `;

    expect(
      await rollupTransform(codeWithoutImport, { include: [/virtual:/] }),
    ).toMatchInlineSnapshot(`
      "const increment = tgpu['~unstable']
              .fn([])(() => {
              });

            console.log(increment);
      "
    `);
  });
});
