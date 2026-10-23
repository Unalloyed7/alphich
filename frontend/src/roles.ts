import type { Role } from './types';

/** Display copy for the two perspectives supported by the analysis API. */
export const roles: Record<Role, { title: string; short: string; focus: string; description: string; priorities: string[] }> = {
  cfo: {
    title: 'Финансовый директор', short: 'CFO', focus: 'Финансовый фокус',
    description: 'Денежные потоки, маржа, стоимость финансирования и финансовое планирование.',
    priorities: ['Денежные потоки', 'Маржа и издержки', 'Финансирование'],
  },
  ceo: {
    title: 'Генеральный директор', short: 'CEO', focus: 'Стратегический фокус',
    description: 'Спрос, инвестиции, изменения на рынках и развитие бизнеса.',
    priorities: ['Спрос и рынки', 'Инвестиции', 'Развитие бизнеса'],
  },
};
